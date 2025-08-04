import asyncio
import aiohttp
import requests
import json
import time
import os
import threading
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import signal
import atexit

# Import g4f API components
import g4f.api

# Detect if running in GitHub Actions
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

@dataclass
class TestResultWithTypes:
    provider: str
    model: str
    working: bool
    response_time: float
    error: str = None
    response_content: str = None
    media_type: str = None
    response_types: List[str] = None

    def __post_init__(self):
        if self.response_types is None:
            self.response_types = ['text']

@dataclass
class TestResult:
    provider: str
    model: str
    working: bool
    response_time: float
    error: str = None
    response_content: str = None
    media_type: str = None

def start_g4f_api_server(port: int = 8081, api_key: str = None):
    """Start g4f API server in a separate thread"""
    def run_server():
        try:
            if api_key:
                g4f.api.AppConfig.set_config(g4f_api_key=api_key)
            g4f.api.run_api(port=port, debug=True)
        except Exception as e:
            print(f"Error starting g4f API server: {e}")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    wait_time = 15 if IS_GITHUB_ACTIONS else 5
    print(f"Starting g4f API server on port {port}...")
    time.sleep(wait_time)
    return server_thread

def cleanup_browsers():
    """Cleanup any remaining browser instances"""
    try:
        try:
            from nodriver import util
            has_nodriver = True
        except ImportError:
            has_nodriver = False

        if has_nodriver:
            for browser in util.get_registered_instances():
                try:
                    if browser.connection:
                        browser.stop()
                        print(f"Stopped browser instance: {browser}")
                except Exception as e:
                    print(f"Error stopping browser: {e}")

            try:
                from g4f.cookies import get_cookies_dir
                lock_file = os.path.join(get_cookies_dir(), ".nodriver_is_open")
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    print("Removed browser lock file")
            except Exception as e:
                print(f"Error removing lock file: {e}")
    except Exception as e:
        print(f"Error during browser cleanup: {e}")

atexit.register(cleanup_browsers)
signal.signal(signal.SIGTERM, lambda signum, frame: cleanup_browsers())
signal.signal(signal.SIGINT, lambda signum, frame: cleanup_browsers())

class ProviderModelFetcherAndTester:
    def __init__(self, base_url: str, api_key: str = None, max_concurrent: int = 100, timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.provider_dir = "provider"
        self.working_dir = "working"
        self.output_dir = "output"
        os.makedirs(self.provider_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.test_messages = [
            {"role": "user", "content": "Hello, are you working? Reply with 'Yes' if you can respond."}
        ]
        self.image_prompt = "a simple test image of a red apple"
        self.video_prompt = "a simple test video of a cat walking"
        self.audio_prompt = "Hello, this is a test audio generation"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def get_model_capabilities(self, session: aiohttp.ClientSession, provider: str, model: str) -> dict:
        """Get model capabilities from the API"""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            async with session.get(
                f"{self.base_url}/api/{provider}/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    models_data = await response.json()
                    if isinstance(models_data, dict) and 'data' in models_data:
                        for model_info in models_data['data']:
                            if model_info.get('id') == model:
                                return {
                                    'image': model_info.get('image', False),
                                    'video': model_info.get('video', False),
                                    'audio': model_info.get('audio', False),
                                    'vision': model_info.get('vision', False)
                                }
        except Exception as e:
            self.logger.debug(f"Error getting model capabilities for {provider}/{model}: {e}")
        return {}

    async def test_provider_model_combination(self, session: aiohttp.ClientSession, provider: str, model: str) -> List[TestResultWithTypes]:
        """Test all capabilities (text, image, audio, video) for a provider-model combination"""
        model_info = await self.get_model_capabilities(session, provider, model)
        results = []
        video_keywords = ['video', 'sora', 'cogvideo', 'mochi', 'hunyuan', 'ltx-video', 'wan2.1']
        audio_keywords = ['audio', 'tts', 'speech', 'voice', 'gtts', 'openai-audio']
        image_keywords = ['flux', 'dall', 'stable', 'image', 'draw', 'paint', 'midjourney', 'diffusion']
        response_types = ['text']
        if model_info.get('image', False) or any(keyword in model.lower() for keyword in image_keywords):
            response_types.append('image')
        if model_info.get('audio', False) or any(keyword in model.lower() for keyword in audio_keywords):
            response_types.append('audio')
        if model_info.get('video', False) or any(keyword in model.lower() for keyword in video_keywords):
            response_types.append('video')
        results.append(await self.test_single_model(session, provider, model, response_types))
        if 'image' in response_types:
            results.append(await self.test_image_generation(session, provider, model, response_types))
        if 'audio' in response_types:
            results.append(await self.test_audio_generation(session, provider, model, response_types))
        if 'video' in response_types:
            results.append(await self.test_video_generation(session, provider, model, response_types))
        return results

    async def test_single_model(self, session: aiohttp.ClientSession, provider: str, model: str, response_types: List[str]) -> TestResultWithTypes:
        """Test a single provider-model combination with streaming"""
        async with self.semaphore:
            start_time = time.time()
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {
                    "model": model,
                    "provider": provider,
                    "messages": self.test_messages,
                    "stream": True,
                    "max_tokens": 50
                }
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_time = time.time() - start_time
                    if response.status == 200:
                        content_parts = []
                        media_responses = []
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith("data: "):
                                    data_str = line_str[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        chunk_data = json.loads(data_str)
                                        if "choices" in chunk_data and chunk_data["choices"]:
                                            choice = chunk_data["choices"][0]
                                            if "delta" in choice and "content" in choice["delta"]:
                                                content = choice["delta"]["content"]
                                                if content:
                                                    content_parts.append(content)
                                            if "message" in choice:
                                                message = choice["message"]
                                                if "audio" in message and message["audio"]:
                                                    await self.save_audio_response(provider, model, message["audio"])
                                                    media_responses.append("audio")
                                                if "images" in message and message["images"]:
                                                    await self.save_image_responses(provider, model, message["images"])
                                                    media_responses.append("images")
                                                if "video" in message and message["video"]:
                                                    await self.save_video_response(provider, model, message["video"])
                                                    media_responses.append("video")
                                    except json.JSONDecodeError:
                                        continue
                        full_content = "".join(content_parts)
                        if full_content:
                            await self.save_text_response(provider, model, full_content)
                        return TestResultWithTypes(
                            provider=provider,
                            model=model,
                            working=True,
                            response_time=response_time,
                            response_content=full_content[:100] if full_content else f"Media: {', '.join(media_responses)}",
                            media_type="text" if full_content else (media_responses[0] if media_responses else None),
                            response_types=response_types
                        )
                    else:
                        error_text = await response.text()
                        return TestResultWithTypes(
                            provider=provider,
                            model=model,
                            working=False,
                            response_time=response_time,
                            error=f"HTTP {response.status}: {error_text[:200]}",
                            response_types=response_types
                        )
            except asyncio.TimeoutError:
                return TestResultWithTypes(
                    provider=provider,
                    model=model,
                    working=False,
                    response_time=self.timeout,
                    error="Request timeout",
                    response_types=response_types
                )
            except Exception as e:
                return TestResultWithTypes(
                    provider=provider,
                    model=model,
                    working=False,
                    response_time=time.time() - start_time,
                    error=f"Unexpected error: {str(e)[:200]}",
                    response_types=response_types
                )

    async def test_image_generation(self, session: aiohttp.ClientSession, provider: str, model: str, response_types: List[str]) -> TestResultWithTypes:
        """Test image generation specifically"""
        async with self.semaphore:
            start_time = time.time()
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {
                    "prompt": self.image_prompt,
                    "model": model,
                    "provider": provider,
                    "response_format": "url",
                    "n": 1
                }
                async with session.post(
                    f"{self.base_url}/v1/images/generate",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_time = time.time() - start_time
                    if response.status == 200:
                        response_data = await response.json()
                        if "data" in response_data and response_data["data"]:
                            image_url = response_data["data"][0].get("url", "")
                            if image_url:
                                await self.save_image_from_url(provider, model, image_url, 0)
                                return TestResultWithTypes(
                                    provider=provider,
                                    model=model,
                                    working=True,
                                    response_time=response_time,
                                    response_content=f"Image generated: {image_url[:50]}...",
                                    media_type="image",
                                    response_types=response_types
                                )
                    error_text = await response.text()
                    return TestResultWithTypes(
                        provider=provider,
                        model=model,
                        working=False,
                        response_time=response_time,
                        error=f"Image generation failed: {error_text[:200]}",
                        media_type="image",
                        response_types=response_types
                    )
            except Exception as e:
                return TestResultWithTypes(
                    provider=provider,
                    model=model,
                    working=False,
                    response_time=time.time() - start_time,
                    error=f"Image generation error: {str(e)[:200]}",
                    media_type="image",
                    response_types=response_types
                )

    async def test_video_generation(self, session: aiohttp.ClientSession, provider: str, model: str, response_types: List[str]) -> TestResultWithTypes:
        """Test video generation specifically"""
        async with self.semaphore:
            start_time = time.time()
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {
                    "prompt": self.video_prompt,
                    "model": model,
                    "provider": provider,
                    "aspect_ratio": "16:9",
                    "duration": 5
                }
                endpoints = [
                    f"{self.base_url}/v1/video/generate",
                    f"{self.base_url}/v1/chat/completions"
                ]
                for endpoint in endpoints:
                    try:
                        if "chat/completions" in endpoint:
                            payload = {
                                "model": model,
                                "provider": provider,
                                "messages": [{"role": "user", "content": self.video_prompt}],
                                "stream": True
                            }
                        async with session.post(
                            endpoint,
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=self.timeout)
                        ) as response:
                            response_time = time.time() - start_time
                            if response.status == 200:
                                if "video/generate" in endpoint:
                                    response_data = await response.json()
                                    if "data" in response_data and response_data["data"]:
                                        video_url = response_data["data"][0].get("url", "")
                                        if video_url:
                                            await self.save_video_from_url(provider, model, video_url)
                                            return TestResultWithTypes(
                                                provider=provider,
                                                model=model,
                                                working=True,
                                                response_time=response_time,
                                                response_content=f"Video generated: {video_url[:50]}...",
                                                media_type="video",
                                                response_types=response_types
                                            )
                                else:
                                    video_found = False
                                    async for line in response.content:
                                        if line:
                                            line_str = line.decode('utf-8').strip()
                                            if line_str.startswith("data: "):
                                                data_str = line_str[6:]
                                                if data_str == "[DONE]":
                                                    break
                                                try:
                                                    chunk_data = json.loads(data_str)
                                                    if "video" in str(chunk_data).lower():
                                                        video_found = True
                                                        break
                                                except json.JSONDecodeError:
                                                    continue
                                    if video_found:
                                        return TestResultWithTypes(
                                            provider=provider,
                                            model=model,
                                            working=True,
                                            response_time=response_time,
                                            response_content="Video response detected",
                                            media_type="video",
                                            response_types=response_types
                                        )
                            break
                    except Exception:
                        continue
                return TestResultWithTypes(
                    provider=provider,
                    model=model,
                    working=False,
                    response_time=response_time,
                    error="Video generation failed on all endpoints",
                    media_type="video",
                    response_types=response_types
                )
            except Exception as e:
                return TestResultWithTypes(
                    provider=provider,
                    model=model,
                    working=False,
                    response_time=time.time() - start_time,
                    error=f"Video generation error: {str(e)[:200]}",
                    media_type="video",
                    response_types=response_types
                )

    async def test_audio_generation(self, session: aiohttp.ClientSession, provider: str, model: str, response_types: List[str]) -> TestResultWithTypes:
        """Test audio/TTS generation specifically"""
        async with self.semaphore:
            start_time = time.time()
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {
                    "input": self.audio_prompt,
                    "model": model,
                    "provider": provider,
                    "voice": "alloy",
                    "response_format": "mp3"
                }
                endpoints = [
                    f"{self.base_url}/v1/audio/speech",
                    f"{self.base_url}/v1/chat/completions"
                ]
                for endpoint in endpoints:
                    try:
                        if "chat/completions" in endpoint:
                            payload = {
                                "model": model,
                                "provider": provider,
                                "messages": [{"role": "user", "content": self.audio_prompt}],
                                "stream": True,
                                "audio": {"voice": "alloy"}
                            }
                        async with session.post(
                            endpoint,
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=self.timeout)
                        ) as response:
                            response_time = time.time() - start_time
                            if response.status == 200:
                                if "audio/speech" in endpoint:
                                    audio_data = await response.read()
                                    if audio_data:
                                        await self.save_audio_from_bytes(provider, model, audio_data)
                                        return TestResultWithTypes(
                                            provider=provider,
                                            model=model,
                                            working=True,
                                            response_time=response_time,
                                            response_content="Audio generated successfully",
                                            media_type="audio",
                                            response_types=response_types
                                        )
                                else:
                                    audio_found = False
                                    async for line in response.content:
                                        if line:
                                            line_str = line.decode('utf-8').strip()
                                            if line_str.startswith("data: "):
                                                data_str = line_str[6:]
                                                if data_str == "[DONE]":
                                                    break
                                                try:
                                                    chunk_data = json.loads(data_str)
                                                    if "choices" in chunk_data and chunk_data["choices"]:
                                                        choice = chunk_data["choices"][0]
                                                        if "message" in choice and "audio" in choice["message"]:
                                                            await self.save_audio_response(provider, model, choice["message"]["audio"])
                                                            audio_found = True
                                                            break
                                                except json.JSONDecodeError:
                                                    continue
                                    if audio_found:
                                        return TestResultWithTypes(
                                            provider=provider,
                                            model=model,
                                            working=True,
                                            response_time=response_time,
                                            response_content="Audio response detected",
                                            media_type="audio",
                                            response_types=response_types
                                        )
                            break
                    except Exception:
                        continue
                return TestResultWithTypes(
                    provider=provider,
                    model=model,
                    working=False,
                    response_time=response_time,
                    error="Audio generation failed on all endpoints",
                    media_type="audio",
                    response_types=response_types
                )
            except Exception as e:
                return TestResultWithTypes(
                    provider=provider,
                    model=model,
                    working=False,
                    response_time=time.time() - start_time,
                    error=f"Audio generation error: {str(e)[:200]}",
                    media_type="audio",
                    response_types=response_types
                )

    async def save_text_response(self, provider: str, model: str, content: str):
        """Save text response to output folder"""
        try:
            safe_model = model.replace('/', '_').replace('\\', '_')
            filename = f"{provider}_{safe_model}_response.txt"
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Text response saved: {filepath}")
        except Exception as e:
            print(f"Error saving text response for {provider}_{model}: {e}")

    async def save_image_from_url(self, provider: str, model: str, image_url: str, index: int):
        """Save image from URL"""
        try:
            safe_model = model.replace('/', '_').replace('\\', '_')
            filename = f"{provider}_{safe_model}_image_{index}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if image_url.startswith(('http://', 'https://')):
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as response:
                        if response.status == 200:
                            with open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            print(f"Image saved: {filepath}")
            elif image_url.startswith('/media/'):
                local_url = f"{self.base_url}{image_url}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(local_url) as response:
                        if response.status == 200:
                            with open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            print(f"Local image saved: {filepath}")
        except Exception as e:
            print(f"Error saving image for {provider}_{model}: {e}")

    async def save_image_responses(self, provider: str, model: str, images: list):
        """Save image responses to output folder"""
        for i, image_url in enumerate(images):
            try:
                safe_model = model.replace('/', '_').replace('\\', '_')
                filename = f"{provider}_{safe_model}_image_{i}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                if image_url.startswith('data:'):
                    import base64
                    base64_data = image_url.split(',')[-1]
                    image_bytes = base64.b64decode(base64_data)
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                    print(f"Image saved: {filepath}")
                elif image_url.startswith(('http://', 'https://')):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as response:
                            if response.status == 200:
                                with open(filepath, 'wb') as f:
                                    async for chunk in response.content.iter_chunked(8192):
                                        f.write(chunk)
                                print(f"Image saved: {filepath}")
            except Exception as e:
                print(f"Error saving image {i} for {provider}_{model}: {e}")

    async def save_video_from_url(self, provider: str, model: str, video_url: str):
        """Save video from URL"""
        try:
            safe_model = model.replace('/', '_').replace('\\', '_')
            filename = f"{provider}_{safe_model}_video.mp4"
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if video_url.startswith(('http://', 'https://')):
                async with aiohttp.ClientSession() as session:
                    async with session.get(video_url) as response:
                        if response.status == 200:
                            with open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            print(f"Video saved: {filepath}")
            elif video_url.startswith('/media/'):
                local_url = f"{self.base_url}{video_url}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(local_url) as response:
                        if response.status == 200:
                            with open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            print(f"Local video saved: {filepath}")
        except Exception as e:
            print(f"Error saving video for {provider}_{model}: {e}")

    async def save_video_response(self, provider: str, model: str, video_data: dict):
        """Save video response to output folder"""
        if 'url' in video_data:
            safe_model = model.replace('/', '_').replace('\\', '_')
            filename = f"{provider}_{safe_model}_video.mp4"
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(video_data['url']) as response:
                        if response.status == 200:
                            with open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            print(f"Video saved: {filepath}")
            except Exception as e:
                print(f"Error saving video for {provider}_{model}: {e}")

    async def save_audio_from_bytes(self, provider: str, model: str, audio_data: bytes):
        """Save audio from raw bytes"""
        try:
            safe_model = model.replace('/', '_').replace('\\', '_')
            filename = f"{provider}_{safe_model}_audio.mp3"
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            print(f"Audio saved: {filepath}")
        except Exception as e:
            print(f"Error saving audio for {provider}_{model}: {e}")

    async def save_audio_response(self, provider: str, model: str, audio_data: dict):
        """Save audio response to output folder"""
        try:
            if isinstance(audio_data, dict) and 'data' in audio_data:
                safe_model = model.replace('/', '_').replace('\\', '_')
                filename = f"{provider}_{safe_model}_audio.mp3"
                filepath = os.path.join(self.output_dir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                if audio_data['data'].startswith('data:'):
                    import base64
                    base64_data = audio_data['data'].split(',')[-1]
                    audio_bytes = base64.b64decode(base64_data)
                    with open(filepath, 'wb') as f:
                        f.write(audio_bytes)
                    print(f"Audio response saved: {filepath}")
                elif audio_data['data'].startswith(('http://', 'https://')):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(audio_data['data']) as response:
                            if response.status == 200:
                                with open(filepath, 'wb') as f:
                                    async for chunk in response.content.iter_chunked(8192):
                                        f.write(chunk)
                                print(f"Audio response saved: {filepath}")
        except Exception as e:
            print(f"Error saving audio for {provider}_{model}: {e}")

    def fetch_providers_and_models(self) -> Dict[str, Any]:
        """Fetch all providers and their models from the g4f API"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        providers_url = f"{self.base_url}/v1/providers"
        try:
            response = requests.get(providers_url, headers=headers)
            response.raise_for_status()
            providers = response.json()
        except requests.RequestException as e:
            print(f"Error fetching providers: {e}")
            return {}
        provider_models = {}
        for provider in providers:
            provider_name = provider.get('id', '')
            if not provider_name:
                continue
            print(f"Fetching models for provider: {provider_name}")
            models_url = f"{self.base_url}/api/{provider_name}/models"
            try:
                response = requests.get(models_url, headers=headers)
                response.raise_for_status()
                models_data = response.json()
                if isinstance(models_data, dict) and 'data' in models_data:
                    models = [model.get('id', '') for model in models_data['data'] if model.get('id')]
                else:
                    models = []
                provider_models[provider_name] = {
                    'provider_info': provider,
                    'models': models,
                    'model_count': len(models)
                }
            except requests.RequestException as e:
                print(f"Error fetching models for {provider_name}: {e}")
                provider_models[provider_name] = {
                    'provider_info': provider,
                    'models': [],
                    'model_count': 0,
                    'error': str(e)
                }
            time.sleep(0.1)
        return provider_models

    def save_to_files(self, data: Dict[str, Any], base_filename: str = "providers_models"):
        """Save the provider-model data to both JSON and TXT formats in provider folder"""
        json_filename = os.path.join(self.provider_dir, f"{base_filename}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {json_filename}")
        txt_filename = os.path.join(self.provider_dir, f"{base_filename}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("PROVIDERS AND THEIR MODELS\n")
            f.write("=" * 50 + "\n\n")
            for provider_name, provider_data in data.items():
                f.write(f"Provider: {provider_name}\n")
                f.write("-" * 30 + "\n")
                provider_info = provider_data.get('provider_info', {})
                f.write(f"URL: {provider_info.get('url', 'N/A')}\n")
                f.write(f"Label: {provider_info.get('label', 'N/A')}\n")
                f.write(f"Model Count: {provider_data.get('model_count', 0)}\n")
                if 'error' in provider_data:
                    f.write(f"Error: {provider_data['error']}\n")
                f.write("Models:\n")
                models = provider_data.get('models', [])
                if models:
                    for model in models:
                        f.write(f"  - {model}\n")
                else:
                    f.write("  No models available\n")
                f.write("\n" + "="*50 + "\n\n")
        print(f"Human-readable data saved to {txt_filename}")

    def create_test_format(self, data: Dict[str, Any], filename: str = "models_for_testing.txt"):
        """Create a simplified format for automated testing in provider folder"""
        test_filename = os.path.join(self.provider_dir, filename)
        with open(test_filename, 'w', encoding='utf-8') as f:
            f.write("# Format: provider_name|model_name\n")
            f.write("# Use this file for automated testing\n\n")
            for provider_name, provider_data in data.items():
                models = provider_data.get('models', [])
                for model in models:
                    f.write(f"{provider_name}|{model}\n")
        print(f"Test format saved to {test_filename}")

    def extract_test_data_from_fetched(self, data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract provider|model pairs from fetched data"""
        provider_model_pairs = []
        for provider_name, provider_data in data.items():
            models = provider_data.get('models', [])
            for model in models:
                provider_model_pairs.append((provider_name, model))
        return provider_model_pairs

    async def test_all_models(self, test_data: List[Tuple[str, str]]) -> List[TestResultWithTypes]:
        """Test all provider-model combinations with controlled concurrency"""
        self.logger.info(f"Starting tests for {len(test_data)} provider-model combinations")
        self.logger.info(f"Max concurrent requests: {self.max_concurrent}")
        self.logger.info(f"Timeout per request: {self.timeout} seconds")
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self.test_provider_model_combination(session, provider, model)
                for provider, model in test_data
            ]
            results = []
            completed = 0
            for coro in asyncio.as_completed(tasks):
                model_results = await coro
                results.extend(model_results)
                completed += 1
                if completed % 50 == 0:
                    working_count = sum(1 for r in results if r.working)
                    self.logger.info(f"Progress: {completed}/{len(test_data)} - Working: {working_count}")
        return results

    async def test_all_models_batched(self, test_data: List[Tuple[str, str]], batch_size: int = 20) -> List[TestResultWithTypes]:
        """Test all models in batches to avoid overwhelming the API"""
        all_results = []
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(test_data) + batch_size - 1)//batch_size
            print(f"Processing batch {batch_num}/{total_batches}")
            batch_results = await self.test_all_models(batch)
            all_results.extend(batch_results)
            if batch_num % 10 == 0:
                print(f"Performing browser cleanup after batch {batch_num}")
                cleanup_browsers()
                await asyncio.sleep(2)
            if i + batch_size < len(test_data):
                await asyncio.sleep(3)
        return all_results

    def save_test_results(self, results: List[TestResultWithTypes], base_filename: str = "test_results"):
        """Save test results to both JSON and TXT formats in working folder"""
        working_results = [r for r in results if r.working]
        non_working_results = [r for r in results if not r.working]
        type_stats = {'text': 0, 'image': 0, 'video': 0, 'audio': 0}
        for result in working_results:
            if result.media_type:
                type_stats[result.media_type] = type_stats.get(result.media_type, 0) + 1
            else:
                type_stats['text'] += 1
        summary = {
            "total_tested": len(results),
            "working_count": len(working_results),
            "non_working_count": len(non_working_results),
            "success_rate": len(working_results) / len(results) * 100 if results else 0,
            "average_response_time": sum(r.response_time for r in working_results) / len(working_results) if working_results else 0,
            "response_type_breakdown": type_stats
        }
        json_data = {
            "summary": summary,
            "working_models": [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "response_time": r.response_time,
                    "response_preview": r.response_content,
                    "media_type": r.media_type or 'text',
                    "response_types": r.response_types
                }
                for r in working_results
            ],
            "non_working_models": [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "error": r.error,
                    "response_time": r.response_time,
                    "expected_response_types": r.response_types
                }
                for r in non_working_results
            ]
        }
        json_filename = os.path.join(self.working_dir, f"{base_filename}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        txt_filename = os.path.join(self.working_dir, f"{base_filename}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("PROVIDER/MODEL TEST RESULTS WITH RESPONSE TYPES\n")
            f.write("=" * 60 + "\n\n")
            f.write("SUMMARY:\n")
            f.write(f"Total Tested: {summary['total_tested']}\n")
            f.write(f"Working: {summary['working_count']}\n")
            f.write(f"Not Working: {summary['non_working_count']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.2f}%\n")
            f.write(f"Average Response Time: {summary['average_response_time']:.2f}s\n\n")
            f.write("RESPONSE TYPE BREAKDOWN:\n")
            for response_type, count in type_stats.items():
                f.write(f"  {response_type.capitalize()}: {count}\n")
            f.write("\n")
            f.write("WORKING MODELS BY RESPONSE TYPE:\n")
            f.write("-" * 40 + "\n")
            for response_type in ['text', 'image', 'video', 'audio']:
                type_results = [r for r in working_results if (r.media_type or 'text') == response_type]
                if type_results:
                    f.write(f"\n{response_type.upper()} MODELS:\n")
                    for result in type_results:
                        f.write(f"  {result.provider}|{result.model} ({result.response_time:.2f}s)\n")
            f.write(f"\nNON-WORKING MODELS:\n")
            f.write("-" * 40 + "\n")
            for result in non_working_results:
                expected_types = ', '.join(result.response_types)
                f.write(f"{result.provider}|{result.model} (Expected: {expected_types}) - Error: {result.error}\n")
        self.save_simple_working_results(results)
        self.logger.info(f"Results saved to {json_filename} and {txt_filename}")
        return summary

    def save_simple_working_results(self, results: List[TestResultWithTypes]):
        """Save simple working results files in working folder"""
        working_results = [r for r in results if r.working]
        working_results_filename = os.path.join(self.working_dir, "working_results.txt")
        with open(working_results_filename, 'w', encoding='utf-8') as f:
            for result in working_results:
                f.write(f"{result.provider}|{result.model}|{result.media_type or 'text'}\n")
        print(f"Simple working results saved to {working_results_filename}")
        models_filename = os.path.join(self.working_dir, "models.txt")
        with open(models_filename, 'w', encoding='utf-8') as f:
            unique_models = list(dict.fromkeys([f"{r.model} ({r.media_type or 'text'})" for r in working_results]))
            for model in unique_models:
                f.write(f"{model}\n")
        print(f"Working models list saved to {models_filename}")

    def fetch_providers_and_models_with_types(self) -> Dict[str, Any]:
        """Fetch all providers and their models from the g4f API with type detection"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        providers_url = f"{self.base_url}/v1/providers"
        try:
            response = requests.get(providers_url, headers=headers)
            response.raise_for_status()
            providers = response.json()
        except requests.RequestException as e:
            print(f"Error fetching providers: {e}")
            return {}
        provider_models = {}
        for provider in providers:
            provider_name = provider.get('id', '')
            if not provider_name:
                continue
            print(f"Fetching models for provider: {provider_name}")
            models_url = f"{self.base_url}/api/{provider_name}/models"
            try:
                response = requests.get(models_url, headers=headers)
                response.raise_for_status()
                models_data = response.json()
                models_with_types = []
                if isinstance(models_data, dict) and 'data' in models_data:
                    for model in models_data['data']:
                        if model.get('id'):
                            model_info = {
                                'id': model.get('id', ''),
                                'image': model.get('image', False),
                                'video': model.get('video', False),
                                'audio': model.get('audio', False),
                                'vision': model.get('vision', False),
                                'response_types': self.determine_response_types(model)
                            }
                            models_with_types.append(model_info)
                provider_models[provider_name] = {
                    'provider_info': provider,
                    'models': models_with_types,
                    'model_count': len(models_with_types)
                }
            except requests.RequestException as e:
                print(f"Error fetching models for {provider_name}: {e}")
                provider_models[provider_name] = {
                    'provider_info': provider,
                    'models': [],
                    'model_count': 0,
                    'error': str(e)
                }
            time.sleep(0.1)
        return provider_models

    def determine_response_types(self, model_info: dict) -> List[str]:
        """Determine what response types a model supports"""
        response_types = ['text']
        if model_info.get('image', False):
            response_types.append('image')
        if model_info.get('video', False):
            response_types.append('video')
        if model_info.get('audio', False):
            response_types.append('audio')
        model_name = model_info.get('id', '').lower()
        if any(keyword in model_name for keyword in ['flux', 'dall', 'stable', 'midjourney', 'diffusion']):
            if 'image' not in response_types:
                response_types.append('image')
        if any(keyword in model_name for keyword in ['video', 'sora', 'cogvideo', 'mochi']):
            if 'video' not in response_types:
                response_types.append('video')
        if any(keyword in model_name for keyword in ['audio', 'tts', 'speech', 'voice']):
            if 'audio' not in response_types:
                response_types.append('audio')
        return response_types

    def save_provider_models_with_types(self, data: Dict[str, Any]):
        """Save provider-model data with response type information"""
        json_filename = os.path.join(self.provider_dir, "provider_models_type.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Provider models with types saved to {json_filename}")
        txt_filename = os.path.join(self.provider_dir, "provider_models_type.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("PROVIDERS AND THEIR MODELS WITH RESPONSE TYPES\n")
            f.write("=" * 60 + "\n\n")
            for provider_name, provider_data in data.items():
                f.write(f"Provider: {provider_name}\n")
                f.write("-" * 40 + "\n")
                provider_info = provider_data.get('provider_info', {})
                f.write(f"URL: {provider_info.get('url', 'N/A')}\n")
                f.write(f"Label: {provider_info.get('label', 'N/A')}\n")
                f.write(f"Model Count: {provider_data.get('model_count', 0)}\n")
                if 'error' in provider_data:
                    f.write(f"Error: {provider_data['error']}\n")
                f.write("Models with Response Types:\n")
                models = provider_data.get('models', [])
                if models:
                    for model in models:
                        model_id = model.get('id', 'Unknown')
                        response_types = ', '.join(model.get('response_types', ['text']))
                        capabilities = []
                        if model.get('image'):
                            capabilities.append('Image')
                        if model.get('video'):
                            capabilities.append('Video')
                        if model.get('audio'):
                            capabilities.append('Audio')
                        if model.get('vision'):
                            capabilities.append('Vision')
                        f.write(f"  - {model_id}\n")
                        f.write(f"    Response Types: {response_types}\n")
                        if capabilities:
                            f.write(f"    Capabilities: {', '.join(capabilities)}\n")
                else:
                    f.write("  No models available\n")
                f.write("\n" + "=" * 60 + "\n\n")
        print(f"Human-readable provider models with types saved to {txt_filename}")

async def main():
    """Main execution function"""
    BASE_URL = "http://localhost:8081"
    API_KEY = "1234"
    start_g4f_api_server(port=8081, api_key=API_KEY)
    tester = ProviderModelFetcherAndTester(BASE_URL, API_KEY, max_concurrent=50, timeout=120)
    print("=== STEP 1: Fetching providers and models ===")
    provider_models_data = tester.fetch_providers_and_models()
    if not provider_models_data:
        print("No data retrieved. Exiting.")
        return
    print(f"Found {len(provider_models_data)} providers")
    tester.save_to_files(provider_models_data)
    tester.create_test_format(provider_models_data)
    total_models = sum(data.get('model_count', 0) for data in provider_models_data.values())
    print(f"\nFetch Summary:")
    print(f"Total providers: {len(provider_models_data)}")
    print(f"Total models: {total_models}")
    print("\n=== STEP 2: Testing all capabilities for provider-model combinations ===")
    test_data = tester.extract_test_data_from_fetched(provider_models_data)
    if not test_data:
        print("No test data available. Exiting.")
        return
    print(f"Testing {len(test_data)} provider-model combinations for all capabilities (text, image, audio, video)")
    start_time = time.time()
    results = await tester.test_all_models_batched(test_data, batch_size=20)
    total_time = time.time() - start_time
    summary = tester.save_test_results(results)
    print(f"\nTesting completed in {total_time:.2f} seconds")
    print(f"Results: {summary['working_count']}/{summary['total_tested']} working ({summary['success_rate']:.2f}%)")
    print(f"Response Type Breakdown: {summary['response_type_breakdown']}")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import aiohttp
import requests
import json
import time
import os
import threading
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import signal
import atexit
import g4f.api

IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

@dataclass
class TestResult:
    provider: str
    model: str
    working: bool
    response_time: float
    error: str = None
    response_content: str = None
    media_type: str = None
    response_types: List[str] = None

    def __post_init__(self):
        self.response_types = self.response_types or ['text']

class ProviderModelTester:
    def __init__(self, base_url: str, api_key: str = None, max_concurrent: int = 50, timeout: int = 120):
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.dirs = {'provider': Path('provider'), 'working': Path('working'), 'output': Path('output')}
        for d in self.dirs.values():
            d.mkdir(exist_ok=True)
        self.test_prompts = {
            'text': [{"role": "user", "content": "Hello, are you working? Reply with 'Yes' if you can respond."}],
            'image': "a simple test image of a red apple",
            'audio': "Hello, this is a test audio generation",
            'video': "a simple test video of a cat walking"
        }
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger(__name__)

    async def _make_request(self, session: aiohttp.ClientSession, endpoints: List[Tuple[str, dict]], media_type: str, response_types: List[str], provider: str, model: str) -> TestResult:
        async with self.semaphore:
            start_time = time.time()
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            for endpoint, payload in endpoints:
                try:
                    async with session.post(f"{self.base_url}/{endpoint}", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                        response_time = time.time() - start_time
                        if response.status != 200:
                            return TestResult(provider, model, False, response_time, f"HTTP {response.status}: {await response.text()[:200]}", media_type=media_type, response_types=response_types)
                        if 'chat/completions' in endpoint and media_type != 'text':
                            return await self._handle_streaming_response(response, provider, model, response_time, response_types, media_type)
                        response_data = await response.json()
                        if "error" in response_data:
                            return TestResult(provider, model, False, response_time, f"API Error: {response_data['error'].get('message', 'Unknown error')[:200]}", media_type=media_type, response_types=response_types)
                        return await self._process_response(response, response_data, provider, model, media_type, response_time, response_types)
                except asyncio.TimeoutError:
                    return TestResult(provider, model, False, self.timeout, "Request timeout", media_type=media_type, response_types=response_types)
                except Exception as e:
                    if endpoint == endpoints[-1][0]:  # Last endpoint
                        return TestResult(provider, model, False, time.time() - start_time, f"Unexpected error: {str(e)[:200]}", media_type=media_type, response_types=response_types)
            return TestResult(provider, model, False, time.time() - start_time, f"{media_type.capitalize()} generation failed on all endpoints", media_type=media_type, response_types=response_types)

    async def _handle_streaming_response(self, response: aiohttp.ClientResponse, provider: str, model: str, response_time: float, response_types: List[str], target_media: str = None) -> TestResult:
        content_parts, media_responses = [], []
        async for line in response.content:
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    try:
                        chunk_data = json.loads(line_str[6:])
                        if "error" in chunk_data:
                            return TestResult(provider, model, False, response_time, f"API Error: {chunk_data['error'].get('message', 'Unknown error')[:200]}", response_types=response_types)
                        if "choices" in chunk_data and chunk_data["choices"]:
                            choice = chunk_data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"] and choice["delta"].get("content"):
                                content_parts.append(choice["delta"]["content"])
                            if "message" in choice:
                                message = choice["message"]
                                for m_type in ['images', 'audio', 'video']:
                                    if m_type in message and message[m_type]:
                                        await self._save_media(m_type if m_type != 'images' else 'image', message[m_type], provider, model)
                                        media_responses.append(m_type)
                    except json.JSONDecodeError:
                        continue
        full_content = "".join(content_parts)
        if target_media and target_media != 'text' and target_media in media_responses:
            return TestResult(provider, model, True, response_time, f"{target_media.capitalize()} response detected", media_type=target_media, response_types=response_types)
        return TestResult(
            provider, model, bool(full_content or media_responses), response_time,
            response_content=full_content[:100] if full_content else f"Media: {', '.join(media_responses)}",
            media_type="text" if full_content else (media_responses[0] if media_responses else None),
            response_types=response_types
        )

    async def _process_response(self, response: aiohttp.ClientResponse, data: dict, provider: str, model: str, media_type: str, response_time: float, response_types: List[str]) -> TestResult:
        if media_type == 'image' and data.get('data'):
            image_url = data["data"][0].get("url", "")
            if image_url:
                await self._save_media('image', image_url, provider, model)
                return TestResult(provider, model, True, response_time, f"Image generated: {image_url[:50]}...", media_type='image', response_types=response_types)
        elif media_type == 'audio' and 'audio/speech' in response.url.path:
            audio_data = await response.read()
            if audio_data:
                await self._save_media('audio', audio_data, provider, model)
                return TestResult(provider, model, True, response_time, "Audio generated successfully", media_type='audio', response_types=response_types)
        elif media_type == 'video' and data.get('data'):
            video_url = data["data"][0].get("url", "")
            if video_url:
                await self._save_media('video', video_url, provider, model)
                return TestResult(provider, model, True, response_time, f"Video generated: {video_url[:50]}...", media_type='video', response_types=response_types)
        return TestResult(provider, model, False, response_time, "No valid data in response", media_type=media_type, response_types=response_types)

    async def _save_media(self, media_type: str, data: any, provider: str, model: str, index: int = 0):
        try:
            safe_model = model.replace('/', '_').replace('\\', '_')
            ext = {'image': 'jpg', 'audio': 'mp3', 'video': 'mp4', 'text': 'txt'}[media_type]
            filename = self.dirs['output'] / f"{provider}_{safe_model}_{media_type}_{index}.{ext}"
            if media_type == 'text':
                with filename.open('w', encoding='utf-8') as f:
                    f.write(data)
            elif isinstance(data, bytes):
                with filename.open('wb') as f:
                    f.write(data)
            elif isinstance(data, str):
                if data.startswith('data:'):
                    import base64
                    with filename.open('wb') as f:
                        f.write(base64.b64decode(data.split(',')[-1]))
                elif data.startswith(('http://', 'https://', '/media/')):
                    url = data if data.startswith(('http://', 'https://')) else f"{self.base_url}{data}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                with filename.open('wb') as f:
                                    async for chunk in resp.content.iter_chunked(8192):
                                        f.write(chunk)
            elif isinstance(data, dict) and 'data' in data:
                await self._save_media(media_type, data['data'], provider, model, index)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    await self._save_media(media_type, item, provider, model, i)
            self.logger.info(f"{media_type.capitalize()} saved: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving {media_type} for {provider}_{model}: {e}")

    async def get_model_capabilities(self, session: aiohttp.ClientSession, provider: str, model: str) -> Tuple[List[str], dict]:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            async with session.get(f"{self.base_url}/api/{provider}/models", headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    models_data = await response.json()
                    for model_info in models_data.get('data', []):
                        if model_info.get('id') == model:
                            return self._determine_response_types(model_info, model), model_info
        except Exception as e:
            self.logger.debug(f"Error getting capabilities for {provider}/{model}: {e}")
        return ['text'], {}

    def _determine_response_types(self, model_info: dict, model_name: str) -> List[str]:
        types = ['text']
        for t in ['image', 'audio', 'video']:
            if model_info.get(t, False) or any(k in model_name.lower() for k in {
                'image': ['flux', 'dall', 'stable', 'midjourney', 'diffusion'],
                'audio': ['audio', 'tts', 'speech', 'voice', 'gtts', 'openai-audio'],
                'video': ['video', 'sora', 'cogvideo', 'mochi', 'hunyuan', 'ltx-video', 'wan2.1']
            }[t]):
                types.append(t)
        return types

    async def test_provider_model(self, session: aiohttp.ClientSession, provider: str, model: str) -> List[TestResult]:
        response_types, model_info = await self.get_model_capabilities(session, provider, model)
        results = []
        endpoints = {
            'text': [('v1/chat/completions', {"model": model, "provider": provider, "messages": self.test_prompts['text'], "stream": True, "max_tokens": 50})],
            'image': [('v1/images/generate', {"prompt": self.test_prompts['image'], "model": model, "provider": provider, "response_format": "url", "n": 1})],
            'audio': [
                ('v1/audio/speech', {"input": self.test_prompts['audio'], "model": model, "provider": provider, "voice": "alloy", "response_format": "mp3"}),
                ('v1/chat/completions', {"model": model, "provider": provider, "messages": [{"role": "user", "content": self.test_prompts['audio']}], "stream": True, "audio": {"voice": "alloy"}})
            ],
            'video': [
                ('v1/video/generate', {"prompt": self.test_prompts['video'], "model": model, "provider": provider, "aspect_ratio": "16:9", "duration": 5}),
                ('v1/chat/completions', {"model": model, "provider": provider, "messages": [{"role": "user", "content": self.test_prompts['video']}], "stream": True})
            ]
        }
        for media_type in response_types:
            results.append(await self._make_request(session, endpoints[media_type], media_type, response_types, provider, model))
        return results

    def fetch_providers_and_models(self) -> Dict[str, dict]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            providers = requests.get(f"{self.base_url}/v1/providers", headers=headers).json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching providers: {e}")
            return {}
        provider_models = {}
        for provider in providers:
            provider_name = provider.get('id', '')
            if not provider_name:
                continue
            try:
                response = requests.get(f"{self.base_url}/api/{provider_name}/models", headers=headers)
                response.raise_for_status()
                models = [{'id': m['id'], 'image': m.get('image', False), 'audio': m.get('audio', False), 'video': m.get('video', False), 'vision': m.get('vision', False)} for m in response.json().get('data', []) if m.get('id')]
                provider_models[provider_name] = {'provider_info': provider, 'models': models, 'model_count': len(models)}
            except requests.RequestException as e:
                self.logger.error(f"Error fetching models for {provider_name}: {e}")
                provider_models[provider_name] = {'provider_info': provider, 'models': [], 'model_count': 0, 'error': str(e)}
            time.sleep(0.1)
        return provider_models

    def save_data(self, data: dict, base_filename: str, dir_key: str, txt_formatter: callable):
        json_file = self.dirs[dir_key] / f"{base_filename}.json"
        txt_file = self.dirs[dir_key] / f"{base_filename}.txt"
        json_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        txt_file.write_text(txt_formatter(data), encoding='utf-8')
        self.logger.info(f"Saved to {json_file} and {txt_file}")

    def _format_providers_txt(self, data: dict) -> str:
        lines = ["PROVIDERS AND MODELS\n" + "=" * 50 + "\n"]
        for p, d in data.items():
            lines.append(f"Provider: {p}\n{'-' * 30}\nURL: {d['provider_info'].get('url', 'N/A')}\n"
                         f"Label: {d['provider_info'].get('label', 'N/A')}\nModel Count: {d['model_count']}\n"
                         f"{'Error: ' + d['error'] + '\n' if 'error' in d else ''}Models:\n")
            for m in d.get('models', []):
                caps = [c for c in ['image', 'audio', 'video', 'vision'] if m.get(c)]
                lines.append(f" - {m['id']}\n   Capabilities: {', '.join(caps) or 'Text'}\n")
            if not d.get('models'):
                lines.append(" No models available\n")
            lines.append(f"\n{'=' * 50}\n")
        return "".join(lines)

    def _format_test_results_txt(self, results: List[TestResult]) -> str:
        working = [r for r in results if r.working]
        non_working = [r for r in results if not r.working]
        type_stats = {t: sum(1 for r in working if (r.media_type or 'text') == t) for t in ['text', 'image', 'video', 'audio']}
        summary = {
            "total_tested": len(results),
            "working_count": len(working),
            "success_rate": len(working) / len(results) * 100 if results else 0,
            "average_response_time": sum(r.response_time for r in working) / len(working) if working else 0,
            "response_type_breakdown": type_stats
        }
        lines = [f"PROVIDER/MODEL TEST RESULTS\n{'=' * 60}\n\nSUMMARY:\nTotal Tested: {summary['total_tested']}\n"
                 f"Working: {summary['working_count']}\nNot Working: {len(non_working)}\nSuccess Rate: {summary['success_rate']:.2f}%\n"
                 f"Average Response Time: {summary['average_response_time']:.2f}s\n\nRESPONSE TYPE BREAKDOWN:\n"
                 + "".join(f" {t.capitalize()}: {c}\n" for t, c in type_stats.items()) + "\nWORKING MODELS BY TYPE:\n" + "-" * 40 + "\n"]
        for t in ['text', 'image', 'video', 'audio']:
            type_results = [r for r in working if (r.media_type or 'text') == t]
            if type_results:
                lines.append(f"\n{t.upper()} MODELS:\n" + "".join(f" {r.provider}|{r.model} ({r.response_time:.2f}s)\n" for r in type_results))
        lines.append(f"\nNON-WORKING MODELS:\n{'-' * 40}\n" + "".join(f"{r.provider}|{r.model} (Expected: {', '.join(r.response_types)}) - Error: {r.error}\n" for r in non_working))
        return "".join(lines)

    def save_test_results(self, results: List[TestResult]):
        working = [r for r in results if r.working]
        data = {
            "summary": {
                "total_tested": len(results),
                "working_count": len(working),
                "non_working_count": len(results) - len(working),
                "success_rate": len(working) / len(results) * 100 if results else 0,
                "average_response_time": sum(r.response_time for r in working) / len(working) if working else 0,
                "response_type_breakdown": {t: sum(1 for r in working if (r.media_type or 'text') == t) for t in ['text', 'image', 'video', 'audio']}
            },
            "working_models": [{"provider": r.provider, "model": r.model, "response_time": r.response_time, "response_preview": r.response_content, "media_type": r.media_type or 'text', "response_types": r.response_types} for r in working],
            "non_working_models": [{"provider": r.provider, "model": r.model, "error": r.error, "response_time": r.response_time, "expected_response_types": r.response_types} for r in results if not r.working]
        }
        self.save_data(data, "test_results", 'working', self._format_test_results_txt)
        with (self.dirs['working'] / "working_results.txt").open('w', encoding='utf-8') as f:
            f.writelines(f"{r.provider}|{r.model}|{r.media_type or 'text'}\n" for r in working)
        with (self.dirs['working'] / "models.txt").open('w', encoding='utf-8') as f:
            f.writelines(f"{m}\n" for m in dict.fromkeys(f"{r.model} ({r.media_type or 'text'})" for r in working))
        return data["summary"]

    async def test_all_models(self, test_data: List[Tuple[str, str]], batch_size: int = 20) -> List[TestResult]:
        results = []
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self.max_concurrent * 2)) as session:
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i + batch_size]
                self.logger.info(f"Processing batch {i // batch_size + 1}/{(len(test_data) + batch_size - 1) // batch_size}")
                tasks = [self.test_provider_model(session, p, m) for p, m in batch]
                for coro in asyncio.as_completed(tasks):
                    results.extend(await coro)
                if i // batch_size % 10 == 0:
                    cleanup_browsers()
                    await asyncio.sleep(2)
                if i + batch_size < len(test_data):
                    await asyncio.sleep(3)
        return results

def start_g4f_api_server(port: int = 8081, api_key: str = None):
    def run_server():
        try:
            if api_key:
                g4f.api.AppConfig.set_config(g4f_api_key=api_key)
            g4f.api.run_api(port=port, debug=True)
        except Exception as e:
            print(f"Error starting g4f API server: {e}")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(15 if IS_GITHUB_ACTIONS else 5)
    return server_thread

def cleanup_browsers():
    try:
        from nodriver import util
        for browser in util.get_registered_instances():
            if browser.connection:
                browser.stop()
        lock_file = Path(g4f.cookies.get_cookies_dir()) / ".nodriver_is_open"
        lock_file.unlink(missing_ok=True)
    except Exception as e:
        print(f"Error during browser cleanup: {e}")

atexit.register(cleanup_browsers)
signal.signal(signal.SIGTERM, lambda _, __: cleanup_browsers())
signal.signal(signal.SIGINT, lambda _, __: cleanup_browsers())

async def main():
    BASE_URL = os.getenv('G4F_BASE_URL', 'http://localhost:8081')
    API_KEY = os.getenv('G4F_API_KEY', '1234')
    start_g4f_api_server(8081, API_KEY)
    tester = ProviderModelTester(BASE_URL, API_KEY)
    provider_models = tester.fetch_providers_and_models()
    if not provider_models:
        print("No data retrieved. Exiting.")
        return
    tester.save_data(provider_models, "providers_models", 'provider', tester._format_providers_txt)
    with (tester.dirs['provider'] / "models_for_testing.txt").open('w', encoding='utf-8') as f:
        f.write("# Format: provider_name|model_name\n")
        test_data = [(p, m['id']) for p, d in provider_models.items() for m in d.get('models', [])]
        f.writelines(f"{p}|{m}\n" for p, m in test_data)
    total_models = sum(d['model_count'] for d in provider_models.values())
    print(f"Found {len(provider_models)} providers, {total_models} models")
    if not test_data:
        print("No test data available. Exiting.")
        return
    start_time = time.time()
    results = await tester.test_all_models(test_data)
    summary = tester.save_test_results(results)
    print(f"Testing completed in {time.time() - start_time:.2f}s\nResults: {summary['working_count']}/{summary['total_tested']} working ({summary['success_rate']:.2f}%)\n"
          f"Response Type Breakdown: {summary['response_type_breakdown']}")

if __name__ == "__main__":
    asyncio.run(main())

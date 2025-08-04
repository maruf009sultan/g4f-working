# 🚀 g4f-working — Your Daily-Updated List of No-Auth Working AI Providers & Models from GPT4Free

[![Daily Provider Model Testing](https://github.com/maruf009sultan/g4f-working/actions/workflows/main.yml/badge.svg)](https://github.com/maruf009sultan/g4f-working/actions/workflows/main.yml)
![g4f-working](https://img.shields.io/badge/API%20KEYS-NOT%20REQUIRED-brightgreen?style=for-the-badge)
![Updates](https://img.shields.io/badge/RESULTS-UPDATED%20DAILY-blue?style=for-the-badge)
![Open Source](https://img.shields.io/badge/OPEN%20SOURCE-YES-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-See%20LICENSE.md-green?style=for-the-badge)

> 💡 **Like this project? Help others discover it — [⭐️ Star the repo](https://github.com/maruf009sultan/g4f-working) to support ongoing updates and visibility!**


---

**g4f-working** is the ultimate, constantly-updated hub for discovering which AI providers and models from [@xtekky/gpt4free](https://github.com/xtekky/gpt4free) are working **right now** — and, crucially, which ones require NO API keys, tokens or cookies.  
Skip the hassle of trial and error. Each day, this project tests all providers and models found in `/provider`, and publicly lists those that work with zero authentication requirements.

---

## 🛠️ How to Use

- **No Python needed. No dependencies. No API keys.**
- **Just open or download `models.txt` or `working_results.txt` for the latest working models.** OR
- **Use [`models.txt`](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/models.txt) for models and [`working_results.txt`](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/working_results.txt) for `Provider | Model | Type` — perfect for automation!**   [UPDATED DAILY]

- **Use the links above to always get the latest results.**

### Example Usage

- **Integrate in your own tools**:  
  Fetch and parse `models.txt` or `working_results.txt` to get today's working options.
- **For automation**:  
  Use a daily cron job, GitHub Action, or simple wget/curl to always stay updated.


---

## ⭐️ Why Use g4f-working?

- **Instant Clarity**: Know at a glance which GPT4Free providers/models are working — no guesswork, no wasted time.
- **No-Auth Only**: We only show providers/models that work WITHOUT any API keys, tokens, or cookies. Plug-and-play!
- **Daily Updates**: Results are refreshed every day, so you always get the latest status.
- **Pure Results**: No Python scripts to run, no setup — just check the result files.
- **Simple Integration**: Use the latest working list in your own projects or tools.

---

## 📂 Repository Structure

| Folder/File              | Description                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| `/provider/`             | All provider and model definitions as per [@xtekky/gpt4free](https://github.com/xtekky/gpt4free).    |
| `/working/`              | All output after daily tests. Contains the latest working models and providers.                      |
| `models.txt`             | [List of all currently working models (name & type only)](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/models.txt).|
| `working_results.txt` | [Provider \| Model \| Type lines for all working, no-auth models](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/working_results.txt) |
| `LICENSE.md`             | The license for this repository. Read for usage/contribution rules.                                   |
| (other files/folders)    | See `/output` for provider/model generated response and `/generated_media` for audio outputs.                      |

---

## ⚡️ How It Works

1. **Scan All Providers/Models**:  
   We scan all available providers and models from `gpt4free` in [@xtekky/gpt4free](https://github.com/xtekky/gpt4free).

2. **Test for No-Auth Access**:  
   Each model/provider is tested to see if it works **without any API key, token, or cookies**.

3. **Log Working Results**:  
   - All working (no-auth) models are logged in `/working/models.txt`.
   - Full results in `/working/working_results.txt` include:  
     `provider|model|type` (e.g., `Blackbox|gpt-4o-mini|text`)

4. **Update Daily**:  
   Results are refreshed every day, so you always see the most up-to-date info.

---

## 📑 File Details

### `models.txt`

- [View raw](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/models.txt)
- **Purpose:**  
  A simple, newline-separated list of all working models (name and type only).  
  Example:
  ```
  gpt-4o-mini|text
  claude-3-haiku|text
  ...
  ```
- **Use case:**  
  Quickly check which models are working with no authentication.

### `working_results.txt`

- [View raw](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/working_results.txt)
- **Purpose:**  
  Detailed results with `provider|model|type` for each working model.
- **Media Type Note:**  
  Some results may appear as text but include media, e.g. audio. Browsers may show media tags like:  
  ```html
  <audio controls src="/media/1754290396_gpt-4o-mini-audio-preview-2024-12-17%2BHello%2C_are_you_working_Reply_with_Yes_if_you_can_respond_2f023c3f300aa51f.mp3"></audio>
  ```
  while the file itself is plain text.  
  These are actual audio outputs, but listed in `.txt` for universal compatibility.

---

## 🔍 What Data Means

- **Provider**: The service/source (from `gpt4free`).
- **Model**: The model name.
- **Type**: Output type (`text`, `image`, `audio`, etc).

E.g.
```
Anthropic|claude-3-5-haiku-latest|text
Blackbox|gpt-4o-mini|audio
```

---

## 🚩 Why Is This Different?

- **No more guessing**: You don’t have to check each provider or model for auth requirements.
- **Zero friction**: Just grab the file, see what’s available, and go.
- **Designed for devs, testers, and enthusiasts**: Anyone who wants to use GPT4Free with maximum simplicity.

---

## 🏛️ Folder-by-Folder Guide

| Folder       | What’s inside                                                                 |
|--------------|-------------------------------------------------------------------------------|
| `/provider`  | Definitions for all providers/models supported by [@xtekky/gpt4free](https://github.com/xtekky/gpt4free). Each file/module is a potential AI source. |
| `/working`   | Daily output: `models.txt` and `working_results.txt` with latest working results. May also include logs or past runs in the future. Nothing here requires code to use. |

---

## ❗️Notes on Media Types

- Sometimes, providers/models output audio or other media.
- The `.txt` result files may show `<audio>` or other tags — these are real, playable outputs (e.g., audio previews), just formatted for browser compatibility.
- Treat all result files as plain text, but expect rich media links or previews inside.

---

## 📈 SEO & Marketing Power

- Optimized for search engines: "no-auth AI", "free AI models", "GPT4Free working list", "daily AI provider status", and more.
- Share and link directly to the raw files for instant access.
- Boost your own project’s reliability by referencing these files.

---

## 💬 FAQ

### Q: What is this project for?
A: For anyone who needs a live, daily-tested list of GPT4Free providers/models that work without API keys, tokens, or cookies.

### Q: Do I need to run code?
A: No! Just use the result files. Everything is pre-generated.

### Q: What if a model outputs audio, but `working_results.txt` is a text file?
A: The results file is always text, but may contain audio links or `<audio>` tags. That’s intentional.

### Q: What’s the difference between `models.txt` and `working_results.txt`?
A: `models.txt` lists working models and types; `working_results.txt` gives you the provider, model, and type — more detail!

### Q: Can I suggest a feature or improvement?
A: Yes! Open an issue or pull request.

---

## 🤝 Contributing

- Pull requests, issues, and suggestions are welcome!
- Make sure to check [LICENSE.md](./LICENSE.md) before contributing.

---

## 📜 License

**This project is licensed as described in [LICENSE.md](./LICENSE.md). Read before use or contribution.**

---

## 🌐 Links

- [@xtekky/gpt4free](https://github.com/xtekky/gpt4free) — the base project for all providers/models.
- [Latest `models.txt`](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/models.txt)
- [Latest `working_results.txt`](https://raw.githubusercontent.com/mojaalagevai/psychic-dollop/refs/heads/main/working/working_results.txt)

---

> _g4f-working: The fastest way to know which GPT4Free providers/models work now — with NO API keys, ever!_

---

![No-Auth AI](https://img.shields.io/badge/NO%20AUTH%20AI-READY-orange?style=for-the-badge&logo=github)

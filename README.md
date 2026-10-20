# 📄 ATS Resume Analyzer

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility and gives specific, actionable improvements, powered by Google's Gemini Flash model.

## Features

- Upload a resume as **PDF or DOCX**
- Optional **job description** input for a targeted keyword-match score
- **ATS score (0-100)** with a weighted breakdown:
  | Category | Weight |
  |---|---|
  | Keywords & Relevance | 30% |
  | Impact & Achievements | 20% |
  | Formatting & Parsability | 20% |
  | Structure & Sections | 15% |
  | Language & Clarity | 15% |
- Strengths, issues, and **missing keywords**
- Prioritised improvements (High / Medium / Low)
- Before/after **bullet point rewrites**
- Download the report as Markdown

> The overall score is calculated in code from the category scores, so it stays consistent. It is an AI estimate, not the output of any specific commercial ATS.

## Setup

**Requirements:** Python 3.10+ and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Provide your API key (pick one)

1. **Secrets file (recommended):** create `.streamlit/secrets.toml`
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
2. **Environment variable:**
   ```bash
   export GEMINI_API_KEY="your-key-here"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
   ```
3. **Paste it in the app's sidebar** at runtime.

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501.

## Choosing the model

The default is `gemini-3.5-flash`. Change it in the sidebar, or set the `GEMINI_MODEL` environment variable. If the name isn't available for your key, the app automatically tries `gemini-3-flash` and then `gemini-2.5-flash`. Model names change over time, so check the [Gemini models page](https://ai.google.dev/gemini-api/docs/models) if you see a "model not found" error.

## Project structure

```
.
├── app.py             # Streamlit UI + Gemini logic
├── requirements.txt   # Python dependencies
├── README.md
└── .gitignore         # keeps secrets out of Git
```

## Troubleshooting

- **`ModuleNotFoundError` on Streamlit Cloud:** the dependency file must be named exactly `requirements.txt` (with an **s**, no spaces) and sit in the repo root next to `app.py`. After fixing it, open **Manage app → Reboot app**.
- **"model not found":** set a current model name in the sidebar (see the Gemini models page above).

## Limitations

- **Scanned/image-only PDFs** can't be read (and neither can most ATS systems). Use a text-based PDF or DOCX.
- Files are limited to 5 MB; very long resumes are truncated to ~20,000 characters.
- Formatting is judged from extracted text only, so visual layout issues (columns, graphics) are inferred, not seen.
- Your resume text is sent to the Gemini API. Don't upload documents you're not comfortable sending to a third-party service.

## Security

Never commit your API key. `.streamlit/secrets.toml` is listed in `.gitignore`. If a key is ever pushed to GitHub, revoke it immediately in Google AI Studio and create a new one.

## License

MIT

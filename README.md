# Eatwise – AI-Powered Nutrition Advisor

A personalized food recommendation and nutritional analysis app built with Streamlit and Azure OpenAI.

## Features

- 🍴 **Get Food Recommendations** – Receive AI-powered meal suggestions tailored to your health goals
- 🔍 **Analyze Nutritional Content** – Upload food photos or describe meals to get detailed nutritional breakdowns
- 💚 **Daily Tips** – Personalized nutrition tips in the sidebar
- 🎯 **Customizable Preferences** – Set health goals, meal types, and dietary restrictions

## Getting Started Locally

### Prerequisites

- Python 3.8+
- Azure OpenAI API credentials (key, endpoint, API version)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Eatwise.git
cd Eatwise
```

2. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Set up local secrets** (important!):
   - Create `.streamlit/secrets.toml` in the project root with your Azure OpenAI credentials:
```toml
AZURE_API_KEY = "your_actual_azure_api_key"
AZURE_ENDPOINT = "https://hkust.azure-api.net"
AZURE_API_VERSION = "2023-05-15"
```
   - **Do NOT commit this file** – it's already in `.gitignore`

5. Run the app locally:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment on Streamlit Cloud

1. Push your repository to GitHub (secrets file is git-ignored, so no credentials are exposed)
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Click "New app" and select your repository
4. Set the main file as `app.py`
5. In app settings, add the following secrets under "Secrets":
   - `AZURE_API_KEY` = your Azure OpenAI API key
   - `AZURE_ENDPOINT` = your Azure endpoint
   - `AZURE_API_VERSION` = `2023-05-15` (or your API version)
6. Click "Deploy"

## Troubleshooting

- **"AZURE_API_KEY is not configured" error**: Make sure `.streamlit/secrets.toml` exists with valid credentials, or set `AZURE_API_KEY` as an environment variable.
- **Streamlit Cloud deployment fails**: Verify that secrets are correctly added in the Streamlit Cloud app settings panel.

## Project Structure

```
Eatwise/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── secrets.toml         # Local secrets (git-ignored)
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Security

- **Never commit secrets** – `.streamlit/secrets.toml` and environment variables are git-ignored
- For local development, use `.streamlit/secrets.toml`
- For Streamlit Cloud, add secrets via the app settings panel
- Rotate API keys regularly if exposed

## Built With

- [Streamlit](https://streamlit.io) – UI framework
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service/) – AI models
- Python libraries: `numpy`, `pandas`, `openai`

## License

MIT License

## Disclaimer

This app provides AI-generated nutritional suggestions for informational purposes only. Always consult with a qualified healthcare professional or registered dietitian before making significant dietary changes.

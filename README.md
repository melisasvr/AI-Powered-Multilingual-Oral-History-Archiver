# Multilingual Oral History Archiver
- The Multilingual Oral History Archiver is a desktop application that enables users to record, upload, transcribe, translate, analyze, and archive oral history stories in a wide range of languages.
- It leverages advanced speech recognition, natural language processing, and translation technologies to make oral history accessible and searchable across linguistic boundaries.

## Features
- Audio Recording & Upload
- Record stories directly via microphone or upload existing audio files.
- Automatic Transcription
- Converts speech to text using Google Speech Recognition with support for multiple languages, including Turkish.
- Translation
- Translates transcriptions into a target language using Google Translate.
- Content Analysis
- Extracts themes and generates tags using NLP for effective categorization and search.
- Quality Scoring
- Evaluates the quality of each transcription based on length, clarity, and completeness.
- Database Archiving
- Stores all oral history records in a local SQLite database with rich metadata.
- Text-to-Speech
- Reads out transcriptions and translations for accessibility.
- User-Friendly GUI
- Intuitive interface built with Tkinter, featuring tabs for recording, browsing, and playback.

## Supported Languages
- English, Turkish, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, and many more.

## Installation
- Prerequisites
- Python 3.8+
- Required Libraries
- Install dependencies using pip:
- `pip install speechrecognition pyttsx3 googletrans==4.0.0-rc1 nltk pyaudio`
- Note:
- For PyAudio installation issues, please take a look at the platform-specific instructions.
- NLTK data (punkt, stopwords) will be downloaded automatically if missing.

## Usage
- Run the Application
- `python multilingual_oral_history_archiver.py`
- Record or Upload Audio
- Use the "Record New Story" tab to record via microphone or upload an audio file.

## Process & Save
- Transcribe, translate, analyze, and save the story to the archive.
- Browse Archive
- Filter and search stories by language, region, or theme.
- Playback & Translate
- Listen to transcriptions or translations using text-to-speech.

## Project Structure
- `multilingual_oral_history_archiver.py` — Main application file containing all core logic and GUI.
- Data Model
- Each oral history record includes:
- Title, Speaker Name, Language, Region, Date, Duration, Audio File Path
- Transcription, Translation, Tags, Themes, Metadata
- Quality Score, Moderation Status

## Customization
- Themes and Tags:
- You can expand the theme_keywords dictionary in ContentAnalyzer to include more domains or keywords.
- Supported Languages:
- To add/remove languages, edit the supported_languages dictionary in LanguageProcessor.

## Troubleshooting
- Microphone/Audio Issues:
- Ensure your microphone is properly configured and accessible to the OS.
- Speech Recognition Errors:
- Audio quality and background noise can affect transcription accuracy.
- Database Errors:
- The SQLite database file (oral_history_archive.db) is created in the working directory.

## Contributing
- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request
  
## Contact:
- For questions or contributions, please open an issue or submit a pull request.

## License
- This project is provided for educational and research purposes. Please check the licenses of third-party libraries used.

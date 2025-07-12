import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import json
import datetime
import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import wave
import pyaudio
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import hashlib

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

@dataclass
class OralHistoryRecord:
    """Data structure for oral history records"""
    id: str
    title: str
    speaker_name: str
    language: str
    region: str
    date_recorded: str
    duration: float
    audio_file_path: str
    transcription: str
    translation: str
    tags: List[str]
    themes: List[str]
    metadata: Dict
    quality_score: float
    moderation_status: str

class LanguageProcessor:
    """Handles language detection, transcription, and translation"""
    
    def __init__(self):
        self.translator = Translator()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Supported languages with Turkish included
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish',  # Turkish language support
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'pl': 'Polish',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'ro': 'Romanian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'mt': 'Maltese',
            'ga': 'Irish',
            'cy': 'Welsh',
            'eu': 'Basque',
            'ca': 'Catalan',
            'gl': 'Galician',
            'uk': 'Ukrainian',
            'be': 'Belarusian',
            'mk': 'Macedonian',
            'sq': 'Albanian',
            'sr': 'Serbian',
            'bs': 'Bosnian',
            'me': 'Montenegrin'
        }
        
        # Turkish-specific language processing
        self.turkish_stopwords = set([
            've', 'bir', 'bu', 'o', 'da', 'de', 'için', 'ile', 'gibi', 'daha',
            'olan', 'olarak', 'var', 'yok', 'çok', 'en', 'hem', 'ya', 'ama',
            'ise', 'eğer', 'şey', 'kadar', 'sonra', 'önce', 'içinde', 'üzerine',
            'altında', 'yanında', 'karşında', 'arasında', 'bunun', 'şunu', 'bunu'
        ])
        
        # Calibrate microphone
        self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                print("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Microphone calibrated.")
        except Exception as e:
            print(f"Warning: Could not calibrate microphone: {e}")
    
    def detect_language(self, text: str) -> str:
        """Detect the language of given text"""
        try:
            detection = self.translator.detect(text)
            return detection.lang if detection.lang in self.supported_languages else 'en'
        except Exception:
            return 'en'  # Default to English
    
    def transcribe_audio(self, audio_file_path: str, language: str = 'auto') -> str:
        """Transcribe audio file to text"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            # Map language codes for speech recognition
            lang_map = {
                'tr': 'tr-TR',  # Turkish
                'en': 'en-US',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'de': 'de-DE',
                'it': 'it-IT',
                'pt': 'pt-BR',
                'ru': 'ru-RU',
                'zh': 'zh-CN',
                'ja': 'ja-JP',
                'ko': 'ko-KR',
                'ar': 'ar-SA',
                'hi': 'hi-IN'
            }
            
            if language == 'auto':
                # Try multiple languages
                for lang_code in ['en-US', 'tr-TR', 'es-ES', 'fr-FR', 'de-DE']:
                    try:
                        text = self.recognizer.recognize_google(audio, language=lang_code)
                        return text
                    except sr.UnknownValueError:
                        continue
                return "Could not transcribe audio"
            else:
                lang_code = lang_map.get(language, 'en-US')
                return self.recognizer.recognize_google(audio, language=lang_code)
                
        except Exception as e:
            return f"Transcription error: {str(e)}"
    
    def record_audio(self, duration: int = 30) -> str:
        """Record audio from microphone"""
        try:
            with self.microphone as source:
                print(f"Recording for {duration} seconds...")
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            
            # Save audio to temporary file
            temp_file = f"temp_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            with open(temp_file, "wb") as f:
                f.write(audio.get_wav_data())
            
            return temp_file
        except Exception as e:
            raise Exception(f"Recording error: {str(e)}")
    
    def translate_text(self, text: str, target_language: str = 'en') -> str:
        """Translate text to target language"""
        try:
            if not text.strip():
                return ""
            
            translation = self.translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            return f"Translation error: {str(e)}"

class ContentAnalyzer:
    """Analyzes content for themes, tags, and categorization"""
    
    def __init__(self):
        self.theme_keywords = {
            'family': ['family', 'mother', 'father', 'children', 'parents', 'siblings', 'grandparents',
                      'aile', 'anne', 'baba', 'çocuk', 'ebeveyn', 'kardeş', 'büyükanne', 'büyükbaba'],
            'war': ['war', 'battle', 'conflict', 'soldier', 'military', 'peace', 'victory', 'defeat',
                   'savaş', 'savaşmak', 'asker', 'askeri', 'barış', 'zafer', 'yenilgi'],
            'immigration': ['immigration', 'migrate', 'moved', 'journey', 'homeland', 'border', 'refugee',
                           'göç', 'göçmen', 'taşınmak', 'yolculuk', 'vatan', 'sınır', 'mülteci'],
            'work': ['work', 'job', 'career', 'profession', 'business', 'employment', 'labor',
                    'iş', 'meslek', 'kariyer', 'işçi', 'çalışmak', 'istihdam'],
            'education': ['school', 'university', 'teacher', 'student', 'learning', 'knowledge', 'study',
                         'okul', 'üniversite', 'öğretmen', 'öğrenci', 'öğrenmek', 'bilgi', 'ders'],
            'culture': ['tradition', 'culture', 'customs', 'heritage', 'religion', 'festival', 'ceremony',
                       'gelenek', 'kültür', 'âdet', 'miras', 'din', 'festival', 'tören'],
            'community': ['community', 'neighborhood', 'village', 'town', 'city', 'neighbors', 'friends',
                         'toplum', 'mahalle', 'köy', 'kasaba', 'şehir', 'komşu', 'arkadaş'],
            'hardship': ['difficult', 'struggle', 'poverty', 'hardship', 'challenge', 'overcome', 'survive',
                        'zor', 'mücadele', 'fakirlik', 'zorluk', 'aşmak', 'hayatta kalmak']
        }
    
    def extract_themes(self, text: str) -> List[str]:
        """Extract themes from text based on keyword analysis"""
        text_lower = text.lower()
        found_themes = []
        
        for theme, keywords in self.theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_themes.append(theme)
        
        return found_themes
    
    def generate_tags(self, text: str, language: str = 'en') -> List[str]:
        """Generate tags from text using NLP"""
        try:
            # Tokenize text
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords
            if language == 'tr':
                stop_words = self.turkish_stopwords
            else:
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    stop_words = set()
            
            # Filter tokens
            filtered_tokens = [token for token in tokens 
                             if token.isalpha() and len(token) > 2 and token not in stop_words]
            
            # Get most common words as tags
            word_freq = Counter(filtered_tokens)
            tags = [word for word, freq in word_freq.most_common(10) if freq > 1]
            
            return tags[:8]  # Return top 8 tags
        except Exception as e:
            print(f"Tag generation error: {e}")
            return []
    
    def calculate_quality_score(self, transcription: str, duration: float) -> float:
        """Calculate quality score based on transcription accuracy and completeness"""
        if not transcription or "error" in transcription.lower():
            return 0.0
        
        # Basic quality metrics
        word_count = len(transcription.split())
        char_count = len(transcription)
        
        # Words per minute (typical speech is 120-150 WPM)
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        # Quality score factors
        length_score = min(1.0, word_count / 50)  # Normalize based on 50 words minimum
        wpm_score = min(1.0, wpm / 150) if wpm > 0 else 0.5
        
        # Check for common transcription errors
        error_indicators = ['[inaudible]', '[unclear]', '...', 'transcription error']
        error_penalty = sum(transcription.lower().count(indicator) for indicator in error_indicators) * 0.1
        
        quality_score = (length_score + wpm_score) / 2 - error_penalty
        return max(0.0, min(1.0, quality_score))

class DatabaseManager:
    """Manages SQLite database for oral history records"""
    
    def __init__(self, db_path: str = "oral_history_archive.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS oral_histories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                speaker_name TEXT NOT NULL,
                language TEXT NOT NULL,
                region TEXT,
                date_recorded TEXT NOT NULL,
                duration REAL,
                audio_file_path TEXT,
                transcription TEXT,
                translation TEXT,
                tags TEXT,
                themes TEXT,
                metadata TEXT,
                quality_score REAL,
                moderation_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_record(self, record: OralHistoryRecord):
        """Save oral history record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO oral_histories 
            (id, title, speaker_name, language, region, date_recorded, duration,
             audio_file_path, transcription, translation, tags, themes, metadata,
             quality_score, moderation_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.id, record.title, record.speaker_name, record.language,
            record.region, record.date_recorded, record.duration,
            record.audio_file_path, record.transcription, record.translation,
            json.dumps(record.tags), json.dumps(record.themes),
            json.dumps(record.metadata), record.quality_score, record.moderation_status
        ))
        
        conn.commit()
        conn.close()
    
    def get_records(self, language: str = None, region: str = None, theme: str = None) -> List[OralHistoryRecord]:
        """Retrieve records with optional filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM oral_histories WHERE 1=1"
        params = []
        
        if language:
            query += " AND language = ?"
            params.append(language)
        
        if region:
            query += " AND region = ?"
            params.append(region)
        
        if theme:
            query += " AND themes LIKE ?"
            params.append(f'%{theme}%')
        
        query += " ORDER BY date_recorded DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            record = OralHistoryRecord(
                id=row[0], title=row[1], speaker_name=row[2], language=row[3],
                region=row[4], date_recorded=row[5], duration=row[6],
                audio_file_path=row[7], transcription=row[8], translation=row[9],
                tags=json.loads(row[10]) if row[10] else [],
                themes=json.loads(row[11]) if row[11] else [],
                metadata=json.loads(row[12]) if row[12] else {},
                quality_score=row[13], moderation_status=row[14]
            )
            records.append(record)
        
        conn.close()
        return records

class TextToSpeechEngine:
    """Handles text-to-speech for playback and accessibility"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.setup_voices()
    
    def setup_voices(self):
        """Setup available voices"""
        self.available_voices = {}
        for voice in self.voices:
            lang = voice.id.split('.')[-1] if '.' in voice.id else 'en'
            self.available_voices[lang] = voice.id
    
    def speak_text(self, text: str, language: str = 'en', rate: int = 200):
        """Convert text to speech"""
        try:
            # Set voice if available
            if language in self.available_voices:
                self.engine.setProperty('voice', self.available_voices[language])
            
            # Set speech rate
            self.engine.setProperty('rate', rate)
            
            # Speak text
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech error: {e}")

class OralHistoryGUI:
    """GUI for the Oral History Archiver"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-Powered Multilingual Oral History Archiver")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.language_processor = LanguageProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.db_manager = DatabaseManager()
        self.tts_engine = TextToSpeechEngine()
        
        # Recording state
        self.is_recording = False
        self.current_record = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Record tab
        self.record_frame = ttk.Frame(notebook)
        notebook.add(self.record_frame, text='Record New Story')
        self.setup_record_tab()
        
        # Browse tab
        self.browse_frame = ttk.Frame(notebook)
        notebook.add(self.browse_frame, text='Browse Archive')
        self.setup_browse_tab()
        
        # Playback tab
        self.playback_frame = ttk.Frame(notebook)
        notebook.add(self.playback_frame, text='Playback & Translate')
        self.setup_playback_tab()
    
    def setup_record_tab(self):
        """Setup recording interface"""
        # Input fields
        info_frame = ttk.LabelFrame(self.record_frame, text="Story Information")
        info_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(info_frame, text="Title:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.title_entry = ttk.Entry(info_frame, width=50)
        self.title_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Speaker Name:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.speaker_entry = ttk.Entry(info_frame, width=50)
        self.speaker_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Language:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.language_var = tk.StringVar(value='tr')  # Default to Turkish
        self.language_combo = ttk.Combobox(info_frame, textvariable=self.language_var, 
                                          values=list(self.language_processor.supported_languages.keys()),
                                          width=47)
        self.language_combo.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Region:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.region_entry = ttk.Entry(info_frame, width=50)
        self.region_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Recording controls
        control_frame = ttk.LabelFrame(self.record_frame, text="Recording Controls")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.record_btn = ttk.Button(control_frame, text="Start Recording", 
                                    command=self.toggle_recording)
        self.record_btn.pack(side='left', padx=5, pady=5)
        
        self.upload_btn = ttk.Button(control_frame, text="Upload Audio File", 
                                    command=self.upload_audio)
        self.upload_btn.pack(side='left', padx=5, pady=5)
        
        self.process_btn = ttk.Button(control_frame, text="Process & Save", 
                                     command=self.process_audio)
        self.process_btn.pack(side='left', padx=5, pady=5)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready to record")
        self.status_label.pack(side='right', padx=5, pady=5)
        
        # Transcription display
        trans_frame = ttk.LabelFrame(self.record_frame, text="Transcription")
        trans_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.transcription_text = scrolledtext.ScrolledText(trans_frame, height=10)
        self.transcription_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_browse_tab(self):
        """Setup archive browsing interface"""
        # Filter controls
        filter_frame = ttk.LabelFrame(self.browse_frame, text="Filters")
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(filter_frame, text="Language:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.filter_language_var = tk.StringVar()
        self.filter_language_combo = ttk.Combobox(filter_frame, textvariable=self.filter_language_var,
                                                 values=[''] + list(self.language_processor.supported_languages.keys()))
        self.filter_language_combo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(filter_frame, text="Region:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.filter_region_entry = ttk.Entry(filter_frame, width=20)
        self.filter_region_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Button(filter_frame, text="Search", command=self.search_records).grid(row=0, column=4, padx=5, pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(self.browse_frame, text="Archive Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview for results
        columns = ('Title', 'Speaker', 'Language', 'Region', 'Date', 'Quality')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Load all records initially
        self.search_records()
    
    def setup_playback_tab(self):
        """Setup playback and translation interface"""
        # Record selection
        select_frame = ttk.LabelFrame(self.playback_frame, text="Select Record")
        select_frame.pack(fill='x', padx=10, pady=5)
        
        self.playback_record_var = tk.StringVar()
        self.playback_combo = ttk.Combobox(select_frame, textvariable=self.playback_record_var, width=80)
        self.playback_combo.pack(side='left', padx=5, pady=5)
        
        ttk.Button(select_frame, text="Load Record", command=self.load_playback_record).pack(side='left', padx=5, pady=5)
        
        # Playback controls
        control_frame = ttk.LabelFrame(self.playback_frame, text="Playback Controls")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Play Original", command=self.play_original).pack(side='left', padx=5, pady=5)
        ttk.Button(control_frame, text="Play Translation", command=self.play_translation).pack(side='left', padx=5, pady=5)
        
        # Translation controls
        ttk.Label(control_frame, text="Translate to:").pack(side='left', padx=5, pady=5)
        self.translate_to_var = tk.StringVar(value='en')
        self.translate_combo = ttk.Combobox(control_frame, textvariable=self.translate_to_var,
                                           values=list(self.language_processor.supported_languages.keys()), width=10)
        self.translate_combo.pack(side='left', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Translate", command=self.translate_current).pack(side='left', padx=5, pady=5)
        
        # Display area
        display_frame = ttk.LabelFrame(self.playback_frame, text="Content Display")
        display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Original text
        ttk.Label(display_frame, text="Original:").pack(anchor='w', padx=5)
        self.original_text = scrolledtext.ScrolledText(display_frame, height=8)
        self.original_text.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Translation
        ttk.Label(display_frame, text="Translation:").pack(anchor='w', padx=5)
        self.translation_text = scrolledtext.ScrolledText(display_frame, height=8)
        self.translation_text.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Update playback combo
        self.update_playback_combo()
    
    def toggle_recording(self):
        """Toggle audio recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.record_btn.config(text="Stop Recording")
        self.status_label.config(text="Recording...")
        
        # Start recording in a separate thread
        threading.Thread(target=self.record_audio_thread, daemon=True).start()
    
    def record_audio_thread(self):
        """Record audio in separate thread"""
        try:
            # Record for 30 seconds or until stopped
            self.current_audio_file = self.language_processor.record_audio(duration=30)
            if self.is_recording:  # Only process if not manually stopped
                self.status_label.config(text="Recording completed")
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            self.status_label.config(text="Recording failed")
        finally:
            self.is_recording = False
            self.record_btn.config(text="Start Recording")
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        self.record_btn.config(text="Start Recording")
        self.status_label.config(text="Recording stopped")
    
    def upload_audio(self):
        """Upload audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.ogg"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_audio_file = file_path
            self.status_label.config(text=f"Audio file loaded: {os.path.basename(file_path)}")
    
    def process_audio(self):
        """Process recorded/uploaded audio"""
        if not hasattr(self, 'current_audio_file') or not self.current_audio_file:
            messagebox.showwarning("No Audio", "Please record or upload an audio file first.")
            return
        
        if not self.title_entry.get().strip():
            messagebox.showwarning("Missing Information", "Please enter a title for the story.")
            return
        
        self.status_label.config(text="Processing audio...")
        
        # Process in separate thread to avoid blocking GUI
        threading.Thread(target=self.process_audio_thread, daemon=True).start()
    
    def process_audio_thread(self):
        """Process audio in separate thread"""
        try:
            # Transcribe audio
            language = self.language_var.get()
            transcription = self.language_processor.transcribe_audio(self.current_audio_file, language)
            
            # Update GUI with transcription
            self.root.after(0, self.transcription_text.delete, '1.0', tk.END)
            self.root.after(0, self.transcription_text.insert, '1.0', transcription)
            
            # Analyze content
            themes = self.content_analyzer.extract_themes(transcription)
            tags = self.content_analyzer.generate_tags(transcription, language)
            
            # Calculate quality score
            audio_duration = self.get_audio_duration(self.current_audio_file)
            quality_score = self.content_analyzer.calculate_quality_score(transcription, audio_duration)
            
            # Translate to English if not already in English
            translation = ""
            if language != 'en':
                translation = self.language_processor.translate_text(transcription, 'en')
            
            # Create record
            record_id = hashlib.md5(f"{self.title_entry.get()}{datetime.datetime.now()}".encode()).hexdigest()
            
            record = OralHistoryRecord(
                id=record_id,
                title=self.title_entry.get().strip(),
                speaker_name=self.speaker_entry.get().strip(),
                language=language,
                region=self.region_entry.get().strip(),
                date_recorded=datetime.datetime.now().isoformat(),
                duration=audio_duration,
                audio_file_path=self.current_audio_file,
                transcription=transcription,
                translation=translation,
                tags=tags,
                themes=themes,
                metadata={
                    'processing_date': datetime.datetime.now().isoformat(),
                    'language_detected': self.language_processor.detect_language(transcription),
                    'word_count': len(transcription.split()),
                    'character_count': len(transcription)
                },
                quality_score=quality_score,
                moderation_status='pending'
            )
            
            # Save to database
            self.db_manager.save_record(record)
            
            # Update GUI
            self.root.after(0, self.status_label.config, {'text': f'Story saved successfully! Quality: {quality_score:.2f}'})
            self.root.after(0, self.search_records)  # Refresh browse tab
            self.root.after(0, self.update_playback_combo)  # Refresh playback tab
            
            # Show success message
            self.root.after(0, messagebox.showinfo, "Success", 
                          f"Story '{record.title}' has been processed and saved!\n\n"
                          f"Themes detected: {', '.join(themes)}\n"
                          f"Quality score: {quality_score:.2f}\n"
                          f"Tags: {', '.join(tags[:5])}")
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Processing Error", str(e))
            self.root.after(0, self.status_label.config, {'text': 'Processing failed'})
    
    def get_audio_duration(self, audio_file_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except:
            return 0.0
    
    def search_records(self):
        """Search and display records in browse tab"""
        try:
            # Get filter values
            language = self.filter_language_var.get() if self.filter_language_var.get() else None
            region = self.filter_region_entry.get().strip() if self.filter_region_entry.get().strip() else None
            
            # Search records
            records = self.db_manager.get_records(language=language, region=region)
            
            # Clear existing items
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Add records to tree
            for record in records:
                self.results_tree.insert('', 'end', values=(
                    record.title,
                    record.speaker_name,
                    self.language_processor.supported_languages.get(record.language, record.language),
                    record.region,
                    record.date_recorded[:10],  # Show date only
                    f"{record.quality_score:.2f}"
                ))
            
            # Update status
            self.root.after(0, lambda: setattr(self, 'search_status', f"Found {len(records)} records"))
            
        except Exception as e:
            messagebox.showerror("Search Error", str(e))
    
    def update_playback_combo(self):
        """Update playback combo with available records"""
        try:
            records = self.db_manager.get_records()
            record_options = [f"{record.title} - {record.speaker_name} ({record.language})" 
                            for record in records]
            self.playback_combo.config(values=record_options)
            
            # Store records for reference
            self.playback_records = records
            
        except Exception as e:
            print(f"Error updating playback combo: {e}")
    
    def load_playback_record(self):
        """Load selected record for playback"""
        try:
            selection = self.playback_record_var.get()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a record to load.")
                return
            
            # Find selected record
            selected_index = self.playback_combo.current()
            if selected_index >= 0 and selected_index < len(self.playback_records):
                self.current_playback_record = self.playback_records[selected_index]
                
                # Display original text
                self.original_text.delete('1.0', tk.END)
                self.original_text.insert('1.0', self.current_playback_record.transcription)
                
                # Display existing translation if available
                self.translation_text.delete('1.0', tk.END)
                if self.current_playback_record.translation:
                    self.translation_text.insert('1.0', self.current_playback_record.translation)
                
                messagebox.showinfo("Record Loaded", 
                                  f"Loaded: {self.current_playback_record.title}\n"
                                  f"Language: {self.current_playback_record.language}\n"
                                  f"Themes: {', '.join(self.current_playback_record.themes)}")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
    
    def play_original(self):
        """Play original transcription using TTS"""
        if not hasattr(self, 'current_playback_record') or not self.current_playback_record:
            messagebox.showwarning("No Record", "Please load a record first.")
            return
        
        try:
            # Play in separate thread to avoid blocking
            threading.Thread(target=self.tts_engine.speak_text, 
                           args=(self.current_playback_record.transcription, 
                                self.current_playback_record.language), 
                           daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))
    
    def play_translation(self):
        """Play translation using TTS"""
        if not hasattr(self, 'current_playback_record') or not self.current_playback_record:
            messagebox.showwarning("No Record", "Please load a record first.")
            return
        
        translation = self.translation_text.get('1.0', tk.END).strip()
        if not translation:
            messagebox.showwarning("No Translation", "Please translate the text first.")
            return
        
        try:
            # Play in separate thread to avoid blocking
            target_lang = self.translate_to_var.get()
            threading.Thread(target=self.tts_engine.speak_text, 
                           args=(translation, target_lang), 
                           daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))
    
    def translate_current(self):
        """Translate current record to selected language"""
        if not hasattr(self, 'current_playback_record') or not self.current_playback_record:
            messagebox.showwarning("No Record", "Please load a record first.")
            return
        
        target_language = self.translate_to_var.get()
        if not target_language:
            messagebox.showwarning("No Target Language", "Please select a target language.")
            return
        
        try:
            # Translate text
            original_text = self.current_playback_record.transcription
            translation = self.language_processor.translate_text(original_text, target_language)
            
            # Display translation
            self.translation_text.delete('1.0', tk.END)
            self.translation_text.insert('1.0', translation)
            
            # Update record with new translation if it's English
            if target_language == 'en':
                self.current_playback_record.translation = translation
                self.db_manager.save_record(self.current_playback_record)
            
        except Exception as e:
            messagebox.showerror("Translation Error", str(e))
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

class ModerationAgent:
    """AI agent for content moderation and quality control"""
    
    def __init__(self):
        self.inappropriate_keywords = {
            'en': ['hate', 'violence', 'explicit', 'inappropriate'],
            'tr': ['nefret', 'şiddet', 'uygunsuz', 'açık saçık']
        }
    
    def moderate_content(self, record: OralHistoryRecord) -> Dict:
        """Moderate content and return moderation result"""
        moderation_result = {
            'status': 'approved',
            'flags': [],
            'confidence': 1.0,
            'review_notes': []
        }
        
        # Check for inappropriate content
        text_to_check = record.transcription.lower()
        keywords = self.inappropriate_keywords.get(record.language, 
                                                  self.inappropriate_keywords['en'])
        
        for keyword in keywords:
            if keyword in text_to_check:
                moderation_result['flags'].append(f"Contains keyword: {keyword}")
                moderation_result['status'] = 'flagged'
                moderation_result['confidence'] -= 0.2
        
        # Check quality score
        if record.quality_score < 0.3:
            moderation_result['flags'].append("Low quality transcription")
            moderation_result['status'] = 'review_needed'
            moderation_result['confidence'] -= 0.3
        
        # Check completeness
        if len(record.transcription) < 50:
            moderation_result['flags'].append("Very short transcription")
            moderation_result['review_notes'].append("Consider if this is a complete story")
        
        return moderation_result

class ArchiveManager:
    """Manages the overall archive system"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.language_processor = LanguageProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.moderation_agent = ModerationAgent()
    
    def batch_process_audio_files(self, audio_files: List[str]) -> List[OralHistoryRecord]:
        """Process multiple audio files in batch"""
        processed_records = []
        
        for audio_file in audio_files:
            try:
                print(f"Processing {audio_file}...")
                
                # Extract metadata from filename if possible
                filename = os.path.basename(audio_file)
                title = filename.replace('.wav', '').replace('.mp3', '').replace('_', ' ')
                
                # Transcribe
                transcription = self.language_processor.transcribe_audio(audio_file)
                
                # Detect language
                language = self.language_processor.detect_language(transcription)
                
                # Analyze content
                themes = self.content_analyzer.extract_themes(transcription)
                tags = self.content_analyzer.generate_tags(transcription, language)
                
                # Calculate quality
                duration = self.get_audio_duration(audio_file)
                quality_score = self.content_analyzer.calculate_quality_score(transcription, duration)
                
                # Create record
                record_id = hashlib.md5(f"{title}{datetime.datetime.now()}".encode()).hexdigest()
                
                record = OralHistoryRecord(
                    id=record_id,
                    title=title,
                    speaker_name="Unknown",
                    language=language,
                    region="Unknown",
                    date_recorded=datetime.datetime.now().isoformat(),
                    duration=duration,
                    audio_file_path=audio_file,
                    transcription=transcription,
                    translation="",
                    tags=tags,
                    themes=themes,
                    metadata={'batch_processed': True},
                    quality_score=quality_score,
                    moderation_status='pending'
                )
                
                # Moderate content
                moderation_result = self.moderation_agent.moderate_content(record)
                record.moderation_status = moderation_result['status']
                
                # Save record
                self.db_manager.save_record(record)
                processed_records.append(record)
                
                print(f"✓ Processed: {title} (Quality: {quality_score:.2f})")
                
            except Exception as e:
                print(f"✗ Error processing {audio_file}: {e}")
        
        return processed_records
    
    def get_audio_duration(self, audio_file_path: str) -> float:
        """Get duration of audio file"""
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except:
            return 0.0
    
    def export_archive(self, output_file: str, format: str = 'json') -> bool:
        """Export archive to different formats"""
        try:
            records = self.db_manager.get_records()
            
            if format == 'json':
                export_data = [asdict(record) for record in records]
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format == 'csv':
                import csv
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Title', 'Speaker', 'Language', 'Region', 
                                   'Date', 'Duration', 'Transcription', 'Translation', 
                                   'Tags', 'Themes', 'Quality Score'])
                    
                    for record in records:
                        writer.writerow([
                            record.id, record.title, record.speaker_name, 
                            record.language, record.region, record.date_recorded,
                            record.duration, record.transcription, record.translation,
                            ', '.join(record.tags), ', '.join(record.themes),
                            record.quality_score
                        ])
            
            return True
            
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get archive statistics"""
        records = self.db_manager.get_records()
        
        stats = {
            'total_records': len(records),
            'languages': {},
            'regions': {},
            'themes': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'total_duration': 0,
            'average_quality': 0
        }
        
        for record in records:
            # Language distribution
            lang_name = self.language_processor.supported_languages.get(record.language, record.language)
            stats['languages'][lang_name] = stats['languages'].get(lang_name, 0) + 1
            
            # Region distribution
            if record.region:
                stats['regions'][record.region] = stats['regions'].get(record.region, 0) + 1
            
            # Theme distribution
            for theme in record.themes:
                stats['themes'][theme] = stats['themes'].get(theme, 0) + 1
            
            # Quality distribution
            if record.quality_score >= 0.7:
                stats['quality_distribution']['high'] += 1
            elif record.quality_score >= 0.4:
                stats['quality_distribution']['medium'] += 1
            else:
                stats['quality_distribution']['low'] += 1
            
            # Duration and quality
            stats['total_duration'] += record.duration
            stats['average_quality'] += record.quality_score
        
        if len(records) > 0:
            stats['average_quality'] /= len(records)
        
        return stats

def main():
    """Main function to run the application"""
    print("Starting AI-Powered Multilingual Oral History Archiver...")
    print("Supported languages include Turkish (tr) and many others!")
    
    # Create and run GUI
    app = OralHistoryGUI()
    app.run()

if __name__ == "__main__":
    main()

# Example usage for batch processing
def batch_process_example():
    """Example of batch processing audio files"""
    archive_manager = ArchiveManager()
    
    # Process multiple audio files
    audio_files = [
        "story1.wav",
        "story2.wav",
        "turkish_story.wav"
    ]
    
    # Note: These files would need to exist in your directory
    # records = archive_manager.batch_process_audio_files(audio_files)
    
    # Export archive
    # archive_manager.export_archive("oral_history_archive.json", "json")
    
    # Get statistics
    stats = archive_manager.get_statistics()
    print("Archive Statistics:")
    print(json.dumps(stats, indent=2))

# Example usage for command line interface
def cli_example():
    """Example command line interface"""
    print("\n=== AI-Powered Multilingual Oral History Archiver ===")
    print("Features:")
    print("- Multi-language support (including Turkish)")
    print("- Real-time transcription and translation")
    print("- Automatic theme detection and tagging")
    print("- Quality scoring and moderation")
    print("- Text-to-speech playback")
    print("- Archive browsing and search")
    print("- Batch processing capabilities")
    print("- Export to JSON/CSV formats")
    print("\nTo run the GUI application, execute: python oral_history_archiver.py")
    print("To process files in batch, use the ArchiveManager class")

if __name__ == "__main__":
    # Run CLI example first, then GUI
    cli_example()
    main()
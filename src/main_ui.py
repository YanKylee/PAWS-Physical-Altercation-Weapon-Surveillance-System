import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
import sys
import json
import shutil
import pygame

# Resolve project root (one level up from src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from human_gate import HumanGate
from weapon_module import WeaponTracker
from altercation_module import AltercationTracker

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_CONFIG_FILE = os.path.join(_ROOT, "config.json")
_AUDIO_DIR = os.path.join(_ROOT, "audio_alerts")
os.makedirs(_AUDIO_DIR, exist_ok=True)
pygame.mixer.init()

_EVIDENCE_DIR = os.path.join(_ROOT, "evidence")
_EVIDENCE_HISTORY = os.path.join(_EVIDENCE_DIR, "history.json")
os.makedirs(_EVIDENCE_DIR, exist_ok=True)
if not os.path.exists(_EVIDENCE_HISTORY):
    with open(_EVIDENCE_HISTORY, "w") as f:
        json.dump([], f)

class PAWSApp(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color="black")
        self.title("PAWS: Automated Surveillance System")
        self.geometry("1100x600")
        
        self.minsize(800, 500)
        
        self.human_gate = HumanGate()
        self.weapon_tracker = WeaponTracker()
        self.altercation_tracker = AltercationTracker()
        
        self.cap = None
        self.is_playing = False
        self.process_thread = None
        self.last_log_time = 0
        self.last_audio_time = 0
        self._current_image = None  # Keep reference to prevent garbage collection flicker
        self._was_people_close = False  # Track proximity transitions
        
        self.config = self._load_config()
        self.audio_files = ["None"] + [f for f in os.listdir(_AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))]
        self.selected_audio = ctk.StringVar(value=self.config.get("last_audio", "None"))
        if self.selected_audio.get() not in self.audio_files:
            self.selected_audio.set("None")
            
        # ===== Main Layout: Tabview =====
        self.tabview = ctk.CTkTabview(self, corner_radius=10, fg_color="black", text_color="white", segmented_button_selected_color="#222222", segmented_button_selected_hover_color="#333333")
        self.tabview.pack(fill="both", expand=True)
        
        self.tab_live = self.tabview.add("Live Surveillance")
        self.tab_history = self.tabview.add("Evidence History")
        
        # --- TAB 1: LIVE SURVEILLANCE ---
        # Enforce strict 3:1 layout ratio
        self.tab_live.grid_columnconfigure(0, weight=3, uniform="main_cols")
        self.tab_live.grid_columnconfigure(1, weight=1, uniform="main_cols")
        self.tab_live.grid_rowconfigure(0, weight=1)
        
        self.left_panel = ctk.CTkFrame(self.tab_live, fg_color="black", corner_radius=0)
        self.left_panel.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # ===== Top Bar =====
        self.top_bar = ctk.CTkFrame(self.left_panel, fg_color="black", corner_radius=0)
        self.top_bar.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        # Brand Title
        self.title_label = ctk.CTkLabel(self.top_bar, text="PAWS", font=ctk.CTkFont(size=24, weight="bold"), text_color="white")
        self.title_label.pack(side="left", padx=(0, 20))
        
        # FPS Badge
        self.fps_badge = ctk.CTkFrame(self.top_bar, fg_color="#111111", corner_radius=4, border_width=1, border_color="#222222")
        self.fps_badge.pack(side="left", padx=(0, 20))
        self.fps_label = ctk.CTkLabel(self.fps_badge, text="FPS: --", text_color="#aaaaaa", font=ctk.CTkFont(size=12, weight="bold"))
        self.fps_label.pack(padx=10, pady=5)

        # Audio Settings Group
        self.audio_frame = ctk.CTkFrame(self.top_bar, fg_color="#111111", corner_radius=4, border_width=1, border_color="#222222")
        self.audio_frame.pack(side="left")
        
        self.audio_label = ctk.CTkLabel(self.audio_frame, text="ALERT SOUND", text_color="#aaaaaa", font=ctk.CTkFont(size=10, weight="bold"))
        self.audio_label.pack(side="left", padx=(10, 5))
        
        self.audio_dropdown = ctk.CTkOptionMenu(
            self.audio_frame, values=self.audio_files, variable=self.selected_audio, command=self._change_audio, 
            width=140, fg_color="#1a1a1a", button_color="#2a2a2a", button_hover_color="#3a3a3a", corner_radius=4
        )
        self.audio_dropdown.pack(side="left", padx=(0, 5), pady=5)
        
        self.add_audio_btn = ctk.CTkButton(self.audio_frame, text="+", width=28, height=28, fg_color="#2a2a2a", hover_color="#3a3a3a", text_color="white", command=self._add_audio)
        self.add_audio_btn.pack(side="left", padx=(0, 5))
        
        self.remove_audio_btn = ctk.CTkButton(self.audio_frame, text="-", width=28, height=28, fg_color="#2a2a2a", hover_color="#3a3a3a", text_color="white", command=self._remove_audio)
        self.remove_audio_btn.pack(side="left", padx=(0, 5))

        # Primary Action Buttons
        self.upload_btn = ctk.CTkButton(
            self.top_bar, text="Insert Video File", fg_color="#ffffff", hover_color="#cccccc", text_color="black", 
            font=ctk.CTkFont(weight="bold"), corner_radius=4, command=self.upload_video
        )
        self.upload_btn.pack(side="right")
        
        self.stop_btn = ctk.CTkButton(
            self.top_bar, text="Stop Video", fg_color="#330000", hover_color="#aa0000", text_color="white", 
            border_width=1, border_color="#aa0000",
            font=ctk.CTkFont(weight="bold"), corner_radius=4, command=self.stop_video, width=80
        )
        self.stop_btn.pack(side="right", padx=(0, 10))
        
        self.video_frame = ctk.CTkFrame(self.left_panel, fg_color="black")
        self.video_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.video_frame.grid_propagate(False) # prevent image from forcing frame size
        self.video_frame.bind("<Configure>", self.on_video_resize)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Video Feed", text_color="#555555", fg_color="black")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.video_width = 640
        self.video_height = 360
        
        # ===== Floating status overlay on top of video =====
        self.status_overlay = ctk.CTkFrame(self.video_frame, fg_color="#111111", corner_radius=6, border_width=1, border_color="#333333")
        self.status_label = ctk.CTkLabel(self.status_overlay, text="", font=ctk.CTkFont(size=13, weight="bold"), text_color="white")
        self.status_label.pack(padx=12, pady=6)
        # Initially hidden — will be placed when needed
        self._status_visible = False
        
        # ===== Right panel: Detection Logs =====
        self.right_panel = ctk.CTkFrame(self.tab_live, fg_color="#050505", corner_radius=0, border_width=1, border_color="#1a1a1a")
        self.right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        self.log_header = ctk.CTkLabel(self.right_panel, text="SYSTEM LOGS", text_color="#888888", font=ctk.CTkFont(size=12, weight="bold"))
        self.log_header.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")
        
        self.log_box = ctk.CTkTextbox(
            self.right_panel, state="disabled", fg_color="#080808", text_color="#00ff00", 
            border_color="#1a1a1a", border_width=1, corner_radius=4, font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.log_box.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

        # --- TAB 2: EVIDENCE HISTORY ---
        self._build_history_tab()

    def _build_history_tab(self):
        """Build the Evidence History tab with pagination and lazy loading."""
        # Clear existing widgets if any
        for widget in self.tab_history.winfo_children():
            widget.destroy()
            
        self.tab_history.grid_rowconfigure(1, weight=1)
        self.tab_history.grid_columnconfigure(0, weight=1)
        
        header_frame = ctk.CTkFrame(self.tab_history, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        title = ctk.CTkLabel(header_frame, text="Evidence History", font=ctk.CTkFont(size=20, weight="bold"), text_color="white")
        title.pack(side="left")
        
        # Count label
        try:
            with open(_EVIDENCE_HISTORY, "r") as f:
                total = len(json.load(f))
        except:
            total = 0
        self._history_count_label = ctk.CTkLabel(header_frame, text=f"  ({total} entries)", text_color="#666666", font=ctk.CTkFont(size=14))
        self._history_count_label.pack(side="left")
        
        # Buttons (right side)
        refresh_btn = ctk.CTkButton(header_frame, text="Refresh", width=90, fg_color="#222222", hover_color="#333333", command=self._build_history_tab)
        refresh_btn.pack(side="right", padx=(5, 0))
        
        clear_btn = ctk.CTkButton(header_frame, text="Clear All", width=90, fg_color="#330000", hover_color="#660000", 
                                  text_color="#ff4444", border_width=1, border_color="#550000", command=self._clear_all_evidence)       
        clear_btn.pack(side="right")
        
        self.history_scroll = ctk.CTkScrollableFrame(self.tab_history, fg_color="#080808")
        self.history_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # Pagination state
        self._history_page_size = 50
        self._history_loaded_count = 0
        self._history_image_refs = []  # Keep references to prevent garbage collection
        
        self._load_history_page()
            
    def _load_history_page(self):
        """Load the next page of evidence entries into the scroll view."""
        try:
            with open(_EVIDENCE_HISTORY, "r") as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            history = []
            
        if not history:
            lbl = ctk.CTkLabel(self.history_scroll, text="No evidence captured yet. Detections will appear here.", text_color="#555555")
            lbl.pack(pady=50)
            return
        
        # Work with newest-first order
        reversed_history = list(reversed(history))
        start = self._history_loaded_count
        end = min(start + self._history_page_size, len(reversed_history))
        page_items = reversed_history[start:end]
        
        if not page_items:
            return
            
        # Remove the existing "Load More" button if present
        if hasattr(self, '_load_more_btn') and self._load_more_btn and self._load_more_btn.winfo_exists():
            self._load_more_btn.destroy()
        
        # Thumbnail cache directory
        thumb_dir = os.path.join(_EVIDENCE_DIR, "thumbnails")
        os.makedirs(thumb_dir, exist_ok=True)
        
        for item in page_items:
            card = ctk.CTkFrame(self.history_scroll, fg_color="#111111", corner_radius=8, border_width=1, border_color="#222222")
            card.pack(fill="x", padx=10, pady=5)
            
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", padx=15, pady=12, fill="y")
            
            time_lbl = ctk.CTkLabel(info_frame, text=item.get("timestamp", "Unknown Time"), font=ctk.CTkFont(size=14, weight="bold"), text_color="white")
            time_lbl.pack(anchor="w")
            
            threats = ", ".join(item.get("threats", []))
            threat_lbl = ctk.CTkLabel(info_frame, text=f"Detected: {threats}", text_color="#ff4444", font=ctk.CTkFont(size=12))
            threat_lbl.pack(anchor="w", pady=(3,0))
            
            img_path = item.get("image_path", "")
            if os.path.exists(img_path):
                try:
                    # Use cached thumbnail if available, otherwise generate one
                    thumb_name = "thumb_" + os.path.basename(img_path)
                    thumb_path = os.path.join(thumb_dir, thumb_name)
                    
                    if os.path.exists(thumb_path):
                        pil_img = Image.open(thumb_path)
                        w_size, base_height = pil_img.size
                    else:
                        pil_img = Image.open(img_path)
                        base_height = 120
                        h_percent = base_height / float(pil_img.size[1])
                        w_size = int(float(pil_img.size[0]) * h_percent)
                        pil_img = pil_img.resize((w_size, base_height), Image.Resampling.LANCZOS)
                        # Cache the thumbnail to disk for fast future loads
                        pil_img.save(thumb_path, "JPEG", quality=80)
                    
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w_size, base_height))
                    self._history_image_refs.append(ctk_img)
                    
                    img_lbl = ctk.CTkLabel(card, image=ctk_img, text="", cursor="hand2")
                    img_lbl.image = ctk_img
                    img_lbl.pack(side="right", padx=15, pady=8)
                    
                    # Click-to-zoom: bind click event to open full image
                    full_path = os.path.abspath(img_path)
                    img_lbl.bind("<Button-1>", lambda e, p=full_path: self._show_zoomed_image(p))
                    
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        self._history_loaded_count = end
        
        # Show "Load More" button if there are remaining entries
        remaining = len(reversed_history) - end
        if remaining > 0:
            self._load_more_btn = ctk.CTkButton(
                self.history_scroll, text=f"Load More ({remaining} remaining)", 
                fg_color="#1a1a1a", hover_color="#2a2a2a", text_color="#aaaaaa",
                border_width=1, border_color="#333333", height=40,
                command=self._load_history_page
            )
            self._load_more_btn.pack(fill="x", padx=10, pady=15)

    def _clear_all_evidence(self):
        """Delete all evidence files and reset the history log."""
        # Confirmation dialog
        confirm = ctk.CTkToplevel(self)
        confirm.title("Confirm Clear All")
        confirm.geometry("360x150")
        confirm.resizable(False, False)
        confirm.transient(self)
        confirm.grab_set()
        confirm.configure(fg_color="#111111")
        
        # Center on parent
        confirm.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 360) // 2
        y = self.winfo_y() + (self.winfo_height() - 150) // 2
        confirm.geometry(f"+{x}+{y}")
        
        lbl = ctk.CTkLabel(confirm, text="Delete all evidence logs and images?\nThis action cannot be undone.", 
                           text_color="white", font=ctk.CTkFont(size=14))
        lbl.pack(pady=(20, 15))
        
        btn_frame = ctk.CTkFrame(confirm, fg_color="transparent")
        btn_frame.pack()
        
        def do_clear():
            confirm.destroy()
            # Delete all evidence image files
            for f in os.listdir(_EVIDENCE_DIR):
                fpath = os.path.join(_EVIDENCE_DIR, f)
                if f.endswith(".jpg"):
                    try:
                        os.remove(fpath)
                    except:
                        pass
            # Delete thumbnail cache
            thumb_dir = os.path.join(_EVIDENCE_DIR, "thumbnails")
            if os.path.exists(thumb_dir):
                shutil.rmtree(thumb_dir, ignore_errors=True)
                os.makedirs(thumb_dir, exist_ok=True)
            # Reset history JSON
            with open(_EVIDENCE_HISTORY, "w") as f:
                json.dump([], f)
            self.log_event("All evidence cleared.")
            self._build_history_tab()
        
        cancel_btn = ctk.CTkButton(btn_frame, text="Cancel", width=100, fg_color="#222222", hover_color="#333333", command=confirm.destroy)
        cancel_btn.pack(side="left", padx=10)
        
        delete_btn = ctk.CTkButton(btn_frame, text="Delete All", width=100, fg_color="#660000", hover_color="#990000", text_color="#ff4444", command=do_clear)
        delete_btn.pack(side="left", padx=10)

    def _show_zoomed_image(self, image_path):
        """Open a popup window showing the full-size evidence image."""
        if not os.path.exists(image_path):
            return
            
        try:
            pil_img = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image for zoom: {e}")
            return
        
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title(f"Evidence — {os.path.basename(image_path)}")
        popup.configure(fg_color="black")
        popup.transient(self)
        
        # Scale image to fit screen (max 90% of screen dimensions)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = int(screen_w * 0.85)
        max_h = int(screen_h * 0.85)
        
        img_w, img_h = pil_img.size
        scale = min(max_w / img_w, max_h / img_h, 1.0)  # Don't upscale
        display_w = int(img_w * scale)
        display_h = int(img_h * scale)
        
        if scale < 1.0:
            pil_img = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        
        popup.geometry(f"{display_w + 20}x{display_h + 20}")
        
        # Center on screen
        popup.update_idletasks()
        x = (screen_w - display_w - 20) // 2
        y = (screen_h - display_h - 20) // 2
        popup.geometry(f"+{x}+{y}")
        
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(display_w, display_h))
        
        img_label = ctk.CTkLabel(popup, image=ctk_img, text="")
        img_label.image = ctk_img  # Keep reference
        img_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Close on Escape key
        popup.bind("<Escape>", lambda e: popup.destroy())
        popup.focus_set()

    def on_video_resize(self, event):
        if event.width > 10 and event.height > 10:
            self.video_width = event.width
            self.video_height = event.height

    def _show_status_overlay(self, text, text_color, bg_color, border_color):
        """Show a floating status badge over the top-right of the video."""
        self.status_label.configure(text=text, text_color=text_color)
        self.status_overlay.configure(fg_color=bg_color, border_color=border_color)
        if not self._status_visible:
            self.status_overlay.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)
            self._status_visible = True

    def _hide_status_overlay(self):
        """Hide the floating status badge."""
        if self._status_visible:
            self.status_overlay.place_forget()
            self._status_visible = False

    def _load_config(self):
        if os.path.exists(_CONFIG_FILE):
            try:
                with open(_CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_config(self):
        with open(_CONFIG_FILE, "w") as f:
            json.dump(self.config, f)

    def _change_audio(self, choice):
        self.config["last_audio"] = choice
        self._save_config()

    def _add_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav;*.ogg;*.m4a")])
        if file_path:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(_AUDIO_DIR, filename)
            shutil.copy(file_path, dest_path)
            
            if filename not in self.audio_files:
                self.audio_files.append(filename)
                self.audio_dropdown.configure(values=self.audio_files)
            
            self.selected_audio.set(filename)
            self._change_audio(filename)
            self.log_event(f"Audio added: {filename}")

    def _remove_audio(self):
        current = self.selected_audio.get()
        if current != "None":
            try:
                os.remove(os.path.join(_AUDIO_DIR, current))
            except Exception as e:
                print(f"Error deleting audio: {e}")
            
            self.audio_files.remove(current)
            self.audio_dropdown.configure(values=self.audio_files)
            self.selected_audio.set("None")
            self._change_audio("None")
            self.log_event(f"Audio removed: {current}")

    def _play_alert_audio(self):
        current_audio = self.selected_audio.get()
        if current_audio == "None":
            return
            
        current_time = time.time()
        # Cooldown: Don't play the sound if it was played in the last 5 seconds
        if current_time - self.last_audio_time > 5.0:
            audio_path = os.path.join(_AUDIO_DIR, current_audio)
            if os.path.exists(audio_path):
                try:
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    self.last_audio_time = current_time
                except Exception as e:
                    print(f"Failed to play audio: {e}")

    def _capture_evidence(self, frame, detections):
        """Saves exactly one clean frame to disk and updates the JSON log."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        display_time = time.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{timestamp_str}_evidence.jpg"
        filepath = os.path.join(_EVIDENCE_DIR, filename)
        
        # Save frame to disk
        cv2.imwrite(filepath, frame)
        
        # Update JSON history
        try:
            with open(_EVIDENCE_HISTORY, "r") as f:
                history = json.load(f)
        except:
            history = []
            
        history.append({
            "timestamp": display_time,
            "threats": list(detections),
            "image_path": filepath
        })
        
        with open(_EVIDENCE_HISTORY, "w") as f:
            json.dump(history, f, indent=4)
            
        self.log_event(f"Evidence captured and saved: {filename}")
        
        # NOTE: We do NOT auto-rebuild the history tab here anymore. 
        # Rebuilding the tab loads and resizes images from disk which freezes the UI thread
        # and murders the video FPS. The user can use the "Refresh Logs" button instead.

    def log_event(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        full_message = f"{timestamp} {message}"
        
        self.log_box.configure(state="normal")
        self.log_box.insert("end", full_message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def stop_video(self):
        if self.is_playing:
            self.is_playing = False
            self.log_event("Stopping video playback...")
            self.after(0, self._hide_status_overlay)

    def upload_video(self):
        if self.is_playing:
            self.stop_video()
            # Wait briefly to ensure the current processing thread terminates
            # and releases the previous video capture completely to avoid crashing.
            time.sleep(0.5)
            
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mpeg;*.mpg;*.mov")])
        if file_path:
            self.log_event("Video inserted. Processing started.")
            self.cap = cv2.VideoCapture(file_path)
            self.is_playing = True
            
            # Reset altercation tracker history for a new video
            self.altercation_tracker.reset()
            
            self.process_thread = threading.Thread(target=self.process_video_thread, daemon=True)
            self.process_thread.start()

    def process_video_thread(self):
        """
        OPTIMIZED processing pipeline with Human Gate.
        Includes frame rate control to prevent blinking.
        """
        # Get video FPS for proper playback speed
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS
        frame_delay = 1.0 / fps
        
        fps_calculation_start = time.time()
        frames_processed = 0
        
        video_start_time = time.time()
        total_frames_processed = 0
        
        while self.is_playing and self.cap.isOpened():
            frame_start = time.time()
            
            success, frame = self.cap.read()
            if success:
                # ===== CAPTURE RAW FRAME FOR EVIDENCE =====
                # Store a clean, high-resolution copy BEFORE downscaling and drawing
                raw_frame = frame.copy()
                
                # Resize frame for processing speed (Standard: 640px width)
                # This significantly boosts FPS on high-res (1080p/4K) video inputs.
                h, w = frame.shape[:2]
                if w > 640:
                    new_h = int(h * (640 / w))
                    frame = cv2.resize(frame, (640, new_h))
                
                # ===== STEP 1: Human Gate (lightweight, runs always) =====
                persons = self.human_gate.detect_humans(frame)
                
                if not persons:
                    # NO humans detected → skip ALL heavy models → save GPU
                    # Reset altercation tracker so next clip starts fresh
                    if len(self.altercation_tracker.frame_buffer) > 0:
                        self.altercation_tracker.reset()
                    self.after(0, self._show_status_overlay,
                              "NO HUMANS DETECTED", "#888888", "#1a1a1a", "#333333")
                    self._display_frame(frame)
                    self._wait_for_frame_rate(frame_start, frame_delay)
                    continue
                
                # Draw human bounding boxes on the frame FIRST (like the old system)
                frame = self.human_gate.draw_human_boxes(frame, persons)
                
                all_detections = []
                
                # ===== STEP 2: Weapon scan (only on detected human crops) =====
                frame, weapon_detections = self.weapon_tracker.scan_frame(frame, persons)
                all_detections.extend(weapon_detections)
                
                # ===== STEP 3: Altercation check =====
                frame, altercation_detections, alt_status = self.altercation_tracker.analyze(frame, persons=persons)
                all_detections.extend(altercation_detections)
                
                # Note: Weapon and Physical Altercation detections are kept strictly independent.
                
                # ===== STEP 4: Update floating status overlay =====
                self._update_overlay_from_status(alt_status, all_detections)
                
                # ===== STEP 5: Threat Triggers (Audio, Logging, Evidence) =====
                if all_detections:
                    self._play_alert_audio()
                    
                    current_time = time.time()
                    # Trigger a log entry (and a picture) at most once every 3 seconds per continuous threat
                    if current_time - self.last_log_time > 3.0:
                        detected_items = ", ".join(all_detections)
                        self.after(0, self.log_event, f"ALERT: {detected_items} detected!")
                        self.last_log_time = current_time
                        
                        # 1 alert from logs = 1 picture only
                        self._capture_evidence(frame, all_detections)
                # (Human bounding boxes are now drawn earlier in the pipeline)
                
                self._display_frame(frame)
                
                # ===== Frame rate control =====
                self._wait_for_frame_rate(frame_start, frame_delay)
                
                # ===== FPS Calculation =====
                frames_processed += 1
                total_frames_processed += 1
                now = time.time()
                elapsed = now - fps_calculation_start
                if elapsed >= 1.0:
                    current_fps = frames_processed / elapsed
                    self.after(0, self._update_fps_label, current_fps)
                    fps_calculation_start = now
                    frames_processed = 0
            else:
                self.is_playing = False
                self.cap.release()
                
                total_time = time.time() - video_start_time
                avg_fps = total_frames_processed / total_time if total_time > 0 else 0.0
                
                self.after(0, self.log_event, f"Video playback finished. Average processing FPS: {avg_fps:.1f}")
                self.after(0, self._update_fps_label, 0.0)
                self.after(0, self._hide_status_overlay)
    
    def _update_overlay_from_status(self, alt_status, all_detections):
        """Map the altercation status dict to a floating overlay badge."""
        status_type = alt_status.get("type", "none")
        confidence = alt_status.get("confidence", 0.0)
        
        if status_type == "alert":
            text = f"⚠ PHYSICAL ALTERCATION ({confidence:.0%})"
            self.after(0, self._show_status_overlay, text, "#ff4444", "#2a0000", "#ff4444")
        elif status_type == "monitoring":
            text = f"MONITORING ({confidence:.0%})"
            self.after(0, self._show_status_overlay, text, "#ffcc00", "#2a2200", "#ffcc00")
        elif status_type == "safe_contact":
            text = f"SAFE INTERACTION"
            self.after(0, self._show_status_overlay, text, "#44cc44", "#002a00", "#44cc44")
        elif all_detections:
            # Weapon-only detections (no altercation status)
            text = f"⚠ {', '.join(all_detections).upper()}"
            self.after(0, self._show_status_overlay, text, "#ff8800", "#2a1500", "#ff8800")
        else:
            text = "NO THREAT DETECTED"
            self.after(0, self._show_status_overlay, text, "#44cc44", "#1a1a1a", "#333333")
    
    def _wait_for_frame_rate(self, frame_start, frame_delay):
        """Sleep to maintain proper video playback speed."""
        elapsed = time.time() - frame_start
        wait_time = frame_delay - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
    
    def _update_fps_label(self, fps_val):
        if fps_val == 0.0:
            self.fps_label.configure(text="FPS: --")
        else:
            self.fps_label.configure(text=f"FPS: {fps_val:.1f}")

    def _display_frame(self, frame):
        """Convert and display a frame on the UI."""
        h, w = frame.shape[:2]
        
        # We subtract 4 to account for subtle padding differences to ensure it doesn't push the frame
        target_w = max(10, self.video_width - 4)
        target_h = max(10, self.video_height - 4)
        
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        resized_frame = cv2.resize(frame, (new_w, new_h))
        color_converted = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)
        
        # Use standard ImageTk to bypass CustomTkinter's internal high-DPI scaling
        # which can cause the video to render larger than its frame and look cropped.
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Keep reference alive to prevent garbage collection flicker
        self._current_image = tk_image
        self.video_label.configure(image=tk_image, text="")

if __name__ == "__main__":
    app = PAWSApp()
    app.log_event("System Initialized.")
    app.mainloop()
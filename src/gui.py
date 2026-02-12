from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .config import OUTPUTS_DIR, PipelineConfig, ensure_layout
from .pipeline import VideoDubPipeline
from .utils import PipelineCancelledError


SOURCE_LANGUAGE_MAP = {
    "Auto Detect": "auto",
    "English": "eng",
    "Urdu": "urd",
}

TARGET_LANGUAGE_MAP = {
    "Pashto": "pbt",
}

PROCESSING_MODE_MAP = {
    "Auto": "auto",
    "CPU": "cpu",
    "GPU": "cuda",
}

VOICE_STYLE_MAP = {
    "Automatic": "auto",
    "Male": "male",
    "Female": "female",
}


@dataclass(slots=True)
class UiJob:
    input_video: Path
    output_root: Path
    status: str = "queued"
    result_path: Path | None = None
    error: str | None = None
    needs_confirmation: bool = False


class DubbingGui:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Pashto Video Dubbing Studio")
        self.root.geometry("1320x880")
        self.root.minsize(1120, 720)
        ensure_layout()

        self.jobs: list[UiJob] = []
        self.worker_thread: threading.Thread | None = None
        self.worker_running = False
        self.queue_stop_requested = False
        self.cancel_event: threading.Event | None = None
        self.current_pipeline: VideoDubPipeline | None = None
        self.current_job: UiJob | None = None

        self.log_queue: Queue[str] = Queue()
        self.progress_queue: Queue[tuple[float, str]] = Queue()

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(OUTPUTS_DIR))
        self.source_language_var = tk.StringVar(value="Auto Detect")
        self.target_language_var = tk.StringVar(value="Pashto")

        self.processing_mode_var = tk.StringVar(value="Auto")
        self.voice_style_var = tk.StringVar(value="Automatic")
        self.verify_var = tk.BooleanVar(value=True)
        self.chunk_var = tk.StringVar(value="20")
        self.model_var = tk.StringVar(value="facebook/seamless-m4t-v2-large")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_percent_var = tk.StringVar(value="0%")
        self.status_var = tk.StringVar(value="Ready")
        self.last_output_var = tk.StringVar(value="N/A")

        self.advanced_visible = False

        self._build_ui()
        self._update_button_states()
        self._poll_queues()

    def _build_ui(self) -> None:
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.main = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.main.pack(fill="both", expand=True, padx=16, pady=14)

        header = ctk.CTkFrame(self.main, corner_radius=14)
        header.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(header, text="Pashto Video Dubbing Studio", font=ctk.CTkFont(size=32, weight="bold")).pack(
            anchor="w",
            padx=18,
            pady=(16, 2),
        )
        ctk.CTkLabel(
            header,
            text="Translate and dub videos with a clean, guided workflow.",
            font=ctk.CTkFont(size=15),
            text_color=("gray30", "gray75"),
        ).pack(anchor="w", padx=18, pady=(0, 16))

        setup_card = ctk.CTkFrame(self.main, corner_radius=14)
        setup_card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(setup_card, text="Video Setup", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            padx=18,
            pady=(14, 8),
        )

        ctk.CTkLabel(setup_card, text="Input Videos").grid(row=1, column=0, padx=(18, 10), pady=8, sticky="w")
        ctk.CTkEntry(setup_card, textvariable=self.input_var, height=36, corner_radius=10).grid(
            row=1,
            column=1,
            columnspan=2,
            padx=(0, 10),
            pady=8,
            sticky="ew",
        )
        ctk.CTkButton(setup_card, text="Browse", width=120, command=self._pick_input).grid(
            row=1,
            column=3,
            padx=(0, 18),
            pady=8,
        )

        ctk.CTkLabel(setup_card, text="Output Folder").grid(row=2, column=0, padx=(18, 10), pady=8, sticky="w")
        ctk.CTkEntry(setup_card, textvariable=self.output_var, height=36, corner_radius=10).grid(
            row=2,
            column=1,
            columnspan=2,
            padx=(0, 10),
            pady=8,
            sticky="ew",
        )
        ctk.CTkButton(setup_card, text="Browse", width=120, command=self._pick_output).grid(
            row=2,
            column=3,
            padx=(0, 18),
            pady=8,
        )

        ctk.CTkLabel(setup_card, text="Source Language").grid(row=3, column=0, padx=(18, 10), pady=(8, 14), sticky="w")
        ctk.CTkOptionMenu(
            setup_card,
            values=list(SOURCE_LANGUAGE_MAP.keys()),
            variable=self.source_language_var,
            height=34,
            corner_radius=10,
        ).grid(row=3, column=1, padx=(0, 10), pady=(8, 14), sticky="w")

        ctk.CTkLabel(setup_card, text="Target Language").grid(row=3, column=2, padx=(10, 10), pady=(8, 14), sticky="w")
        ctk.CTkOptionMenu(
            setup_card,
            values=list(TARGET_LANGUAGE_MAP.keys()),
            variable=self.target_language_var,
            height=34,
            corner_radius=10,
        ).grid(row=3, column=3, padx=(0, 18), pady=(8, 14), sticky="w")

        setup_card.grid_columnconfigure(1, weight=1)
        setup_card.grid_columnconfigure(2, weight=1)

        self.actions_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.actions_frame.pack(fill="x", pady=(0, 10))

        self.start_btn = ctk.CTkButton(self.actions_frame, text="Start", width=130, command=self._start_worker)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.open_output_btn = ctk.CTkButton(
            self.actions_frame,
            text="Open Output Folder",
            width=160,
            command=self._open_output_folder,
        )
        self.open_output_btn.pack(side="left", padx=(0, 8))
        self.advanced_btn = ctk.CTkButton(
            self.actions_frame,
            text="Show Advanced Settings",
            width=190,
            command=self._toggle_advanced,
            fg_color=("gray65", "gray30"),
            hover_color=("gray55", "gray25"),
        )
        self.advanced_btn.pack(side="left", padx=8)

        self.advanced_card = ctk.CTkFrame(self.main, corner_radius=14)
        self.advanced_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(self.advanced_card, text="Advanced Settings", font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            padx=18,
            pady=(12, 8),
        )

        ctk.CTkLabel(self.advanced_card, text="Processing Mode").grid(row=1, column=0, padx=(18, 10), pady=8, sticky="w")
        ctk.CTkOptionMenu(
            self.advanced_card,
            values=list(PROCESSING_MODE_MAP.keys()),
            variable=self.processing_mode_var,
            height=34,
            corner_radius=10,
        ).grid(row=1, column=1, padx=(0, 10), pady=8, sticky="w")

        ctk.CTkLabel(self.advanced_card, text="Voice Style").grid(row=1, column=2, padx=(10, 10), pady=8, sticky="w")
        ctk.CTkOptionMenu(
            self.advanced_card,
            values=list(VOICE_STYLE_MAP.keys()),
            variable=self.voice_style_var,
            height=34,
            corner_radius=10,
        ).grid(row=1, column=3, padx=(0, 18), pady=8, sticky="w")

        ctk.CTkCheckBox(
            self.advanced_card,
            text="Enable Translation Quality Check",
            variable=self.verify_var,
            onvalue=True,
            offvalue=False,
        ).grid(row=2, column=0, columnspan=2, padx=(18, 10), pady=8, sticky="w")

        ctk.CTkLabel(self.advanced_card, text="Segment Length (seconds)").grid(
            row=2,
            column=2,
            padx=(10, 10),
            pady=8,
            sticky="w",
        )
        ctk.CTkEntry(self.advanced_card, textvariable=self.chunk_var, width=120, height=34, corner_radius=10).grid(
            row=2,
            column=3,
            padx=(0, 18),
            pady=8,
            sticky="w",
        )

        ctk.CTkLabel(self.advanced_card, text="Translation Model").grid(
            row=3,
            column=0,
            padx=(18, 10),
            pady=(8, 14),
            sticky="w",
        )
        ctk.CTkEntry(self.advanced_card, textvariable=self.model_var, height=34, corner_radius=10).grid(
            row=3,
            column=1,
            columnspan=3,
            padx=(0, 18),
            pady=(8, 14),
            sticky="ew",
        )
        self.advanced_card.grid_columnconfigure(1, weight=1)
        self.advanced_card.grid_columnconfigure(3, weight=1)
        self.advanced_card.pack_forget()

        queue_card = ctk.CTkFrame(self.main, corner_radius=14)
        queue_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(queue_card, text="Queue", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=18, pady=(12, 8))
        self.queue_box = ctk.CTkTextbox(queue_card, height=120, corner_radius=10)
        self.queue_box.pack(fill="x", padx=18, pady=(0, 14))
        self.queue_box.configure(state="disabled")

        progress_card = ctk.CTkFrame(self.main, corner_radius=14)
        progress_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(progress_card, text="Progress", font=ctk.CTkFont(size=18, weight="bold")).pack(
            anchor="w",
            padx=18,
            pady=(12, 8),
        )
        status_row = ctk.CTkFrame(progress_card, fg_color="transparent")
        status_row.pack(fill="x", padx=18)
        ctk.CTkLabel(status_row, textvariable=self.status_var, font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkLabel(status_row, textvariable=self.progress_percent_var, font=ctk.CTkFont(size=24, weight="bold")).pack(
            side="right"
        )
        self.progress = ctk.CTkProgressBar(progress_card, variable=self.progress_var, height=22, corner_radius=10)
        self.progress.pack(fill="x", padx=18, pady=(8, 10))
        self.progress.set(0.0)

        self.cancel_btn = ctk.CTkButton(
            progress_card,
            text="Cancel Current Process",
            width=220,
            command=self._cancel_current,
            fg_color="#C62828",
            hover_color="#B71C1C",
        )
        self.cancel_btn.pack(padx=18, pady=(0, 14), anchor="w")

        output_row = ctk.CTkFrame(self.main, fg_color="transparent")
        output_row.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(output_row, text="Last Output File").pack(side="left", padx=(2, 10))
        ctk.CTkEntry(output_row, textvariable=self.last_output_var, height=34, corner_radius=10).pack(
            side="left",
            fill="x",
            expand=True,
        )
        self.open_file_btn = ctk.CTkButton(output_row, text="Open File", width=120, command=self._open_last_output_file)
        self.open_file_btn.pack(side="left", padx=(10, 0))

        logs_card = ctk.CTkFrame(self.main, corner_radius=14)
        logs_card.pack(fill="both", expand=True)
        ctk.CTkLabel(logs_card, text="Logs", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=18, pady=(12, 8))
        self.log_box = ctk.CTkTextbox(logs_card, corner_radius=10)
        self.log_box.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        self.log_box.configure(state="disabled")

    def _toggle_advanced(self) -> None:
        if self.advanced_visible:
            self.advanced_card.pack_forget()
            self.advanced_visible = False
            self.advanced_btn.configure(text="Show Advanced Settings")
            return
        self.advanced_card.pack(fill="x", pady=(0, 12), after=self.actions_frame)
        self.advanced_visible = True
        self.advanced_btn.configure(text="Hide Advanced Settings")

    def _update_button_states(self) -> None:
        running = bool(self.worker_running)
        self.start_btn.configure(state=("disabled" if running else "normal"))
        self.cancel_btn.configure(state=("normal" if running else "disabled"))

    def _log(self, message: str) -> None:
        self.log_queue.put(message)

    def _progress(self, value: float, message: str) -> None:
        self.progress_queue.put((max(0.0, min(1.0, value)), message))

    def _pick_input(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select up to 3 videos",
            filetypes=[("Video files", "*.mp4;*.mkv;*.mov;*.avi"), ("All files", "*.*")],
        )
        if not selected:
            return
        selected_paths = [Path(p).resolve() for p in selected]
        if len(selected_paths) > 3:
            messagebox.showerror("Selection Limit", "Please select a maximum of 3 videos at a time.")
            return
        self._queue_selected_videos(selected_paths)

    def _pick_output(self) -> None:
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_var.set(selected)

    def _open_output_folder(self) -> None:
        out_dir = Path(self.output_var.get().strip() or str(OUTPUTS_DIR))
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out_dir))
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open folder: {exc}")

    def _open_last_output_file(self) -> None:
        path = Path(self.last_output_var.get().strip())
        if not path.exists():
            messagebox.showerror("Missing file", "No completed output file found yet.")
            return
        try:
            os.startfile(str(path))
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open file: {exc}")

    def _queue_selected_videos(self, paths: list[Path]) -> None:
        output_root = Path(self.output_var.get().strip() or str(OUTPUTS_DIR))
        output_root.mkdir(parents=True, exist_ok=True)

        existing = {job.input_video.resolve() for job in self.jobs}
        added: list[Path] = []
        for path in paths:
            if not path.exists():
                continue
            if path in existing:
                continue
            needs_confirmation = bool(self.worker_running)
            self.jobs.append(UiJob(input_video=path, output_root=output_root, needs_confirmation=needs_confirmation))
            existing.add(path)
            added.append(path)

        if not added:
            messagebox.showinfo("Queue", "Selected videos are already in queue.")
            return

        if len(added) == 1:
            self.input_var.set(str(added[0]))
        else:
            self.input_var.set(f"{len(added)} videos selected")

        self._refresh_queue()
        for path in added:
            if self.worker_running:
                self._log(f"Queued (awaiting confirmation after current run): {path}")
            else:
                self._log(f"Queued: {path}")

    def _render_queue_text(self) -> str:
        if not self.jobs:
            return "No jobs in queue."
        lines: list[str] = []
        for idx, job in enumerate(self.jobs, start=1):
            suffix = f" -> {job.result_path}" if job.result_path else ""
            error = f" [ERROR: {job.error}]" if job.error else ""
            if job.status == "queued" and job.needs_confirmation:
                lines.append(f"{idx}. {job.input_video.name} [queued - waiting approval]{suffix}{error}")
            else:
                lines.append(f"{idx}. {job.input_video.name} [{job.status}]{suffix}{error}")
        return "\n".join(lines)

    def _refresh_queue(self) -> None:
        self.queue_box.configure(state="normal")
        self.queue_box.delete("1.0", "end")
        self.queue_box.insert("1.0", self._render_queue_text())
        self.queue_box.configure(state="disabled")

    def _refresh_queue_async(self) -> None:
        self.root.after(0, self._refresh_queue)

    def _set_last_output_async(self, value: str) -> None:
        self.root.after(0, lambda: self.last_output_var.set(value))

    def _make_pipeline(self, cancel_event: threading.Event) -> VideoDubPipeline:
        source_lang = SOURCE_LANGUAGE_MAP.get(self.source_language_var.get(), "auto")
        target_lang = TARGET_LANGUAGE_MAP.get(self.target_language_var.get(), "pbt")
        processing_mode = PROCESSING_MODE_MAP.get(self.processing_mode_var.get(), "auto")
        voice_style = VOICE_STYLE_MAP.get(self.voice_style_var.get(), "auto")
        try:
            chunk_seconds = int((self.chunk_var.get() or "20").strip())
        except ValueError:
            chunk_seconds = 20
            self.chunk_var.set("20")

        cfg = PipelineConfig(
            model_name=self.model_var.get().strip() or "facebook/seamless-m4t-v2-large",
            source_lang=source_lang,
            target_lang=target_lang,
            device_policy=processing_mode,
            chunk_seconds=chunk_seconds,
            tts_gender_mode=voice_style,
            enable_translation_verification=bool(self.verify_var.get()),
        )
        return VideoDubPipeline(cfg, log_fn=self._log, cancel_event=cancel_event)

    def _start_worker(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Queue Running", "Queue is already running.")
            return
        pending = [job for job in self.jobs if job.status in {"queued", "failed"}]
        if not pending:
            messagebox.showerror("No Jobs", "Please select at least one video first.")
            return
        self.queue_stop_requested = False
        self.worker_running = True
        self._update_button_states()
        self.worker_thread = threading.Thread(target=self._worker_main, daemon=True)
        self.worker_thread.start()

    def _cancel_current(self) -> None:
        if not self.worker_running:
            messagebox.showinfo("No Active Process", "Nothing is currently running.")
            return
        confirmed = messagebox.askyesno(
            "Cancel Current Process",
            "Are you sure you want to cancel now?\n\nPartial files for the current job will be deleted.",
        )
        if not confirmed:
            return
        self.queue_stop_requested = True
        if self.cancel_event is not None:
            self.cancel_event.set()
        if self.current_pipeline is not None:
            self.current_pipeline.request_cancel()
        self._progress(float(self.progress_var.get()), "Cancelling...")
        self._log("Cancellation requested by user.")

    def _ask_yes_no_from_worker(self, title: str, prompt: str) -> bool:
        result = {"value": False}
        done = threading.Event()

        def _prompt() -> None:
            result["value"] = messagebox.askyesno(title, prompt)
            done.set()

        self.root.after(0, _prompt)
        done.wait()
        return bool(result["value"])

    def _worker_main(self) -> None:
        for job in self.jobs:
            if self.queue_stop_requested:
                break
            if job.status in {"done", "running", "cancelled", "skipped"}:
                continue
            if job.needs_confirmation:
                approved = self._ask_yes_no_from_worker(
                    "Start next video?",
                    f"Current job is complete.\n\nStart next video now?\n{job.input_video.name}",
                )
                if not approved:
                    job.status = "skipped"
                    job.error = "Skipped by user"
                    self._log(f"Skipped by user choice: {job.input_video.name}")
                    self._refresh_queue_async()
                    continue
                job.needs_confirmation = False
                self._refresh_queue_async()
            try:
                self.cancel_event = threading.Event()
                self.current_pipeline = self._make_pipeline(cancel_event=self.cancel_event)
                self.current_job = job
                job.error = None
                job.status = "running"
                self._refresh_queue_async()
                self._progress(0.0, f"Processing {job.input_video.name}")
                result = self.current_pipeline.run(job.input_video, job.output_root, progress_fn=self._progress)
                job.status = "done"
                job.result_path = result.final_video
                self._set_last_output_async(str(result.final_video))
                self._log(f"Completed: {result.final_video}")
            except PipelineCancelledError:
                job.status = "cancelled"
                job.error = "Cancelled by user"
                self.queue_stop_requested = True
                self._log(f"Cancelled: {job.input_video.name}. Partial files were cleaned.")
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                self._log(f"Failed {job.input_video.name}: {exc}")
            finally:
                self.current_pipeline = None
                self.current_job = None
                self.cancel_event = None
                self._refresh_queue_async()
                self._progress(0.0, "Ready")

        self.worker_running = False
        self._log("Queue finished.")

    def _poll_queues(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", message + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except Empty:
            pass

        try:
            while True:
                value, message = self.progress_queue.get_nowait()
                self.progress_var.set(value)
                self.progress.set(value)
                self.status_var.set(message)
                self.progress_percent_var.set(f"{int(round(value * 100))}%")
        except Empty:
            pass

        self._update_button_states()
        self.root.after(100, self._poll_queues)


def launch_gui() -> None:
    root = ctk.CTk()
    app = DubbingGui(root)
    app._log("Ready. Select videos and click Start.")
    root.mainloop()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import OUTPUTS_DIR, PipelineConfig, ensure_layout
from .pipeline import VideoDubPipeline


@dataclass(slots=True)
class UiJob:
    input_video: Path
    output_root: Path
    status: str = "queued"
    result_path: Path | None = None
    error: str | None = None


class DubbingGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pashto Video Dubbing + Lip Sync (Windows)")
        self.root.geometry("1080x760")
        ensure_layout()

        self.jobs: list[UiJob] = []
        self.worker_thread: threading.Thread | None = None
        self.stop_requested = False
        self.log_queue: Queue[str] = Queue()
        self.progress_queue: Queue[tuple[float, str]] = Queue()

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(OUTPUTS_DIR))
        self.device_var = tk.StringVar(value="auto")
        self.source_lang_var = tk.StringVar(value="auto")
        self.model_var = tk.StringVar(value="facebook/seamless-m4t-v2-large")
        self.target_lang_var = tk.StringVar(value="pbt")
        self.voice_gender_var = tk.StringVar(value="auto")
        self.verify_var = tk.BooleanVar(value=True)
        self.chunk_var = tk.IntVar(value=20)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Idle")
        self.last_output_var = tk.StringVar(value="N/A")

        self._build_ui()
        self._poll_queues()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        row1 = ttk.Frame(frm)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Input video").pack(side="left")
        ttk.Entry(row1, textvariable=self.input_var, width=90).pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(row1, text="Browse", command=self._pick_input).pack(side="left")

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Output folder").pack(side="left")
        ttk.Entry(row2, textvariable=self.output_var, width=88).pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(row2, text="Browse", command=self._pick_output).pack(side="left")

        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="Device").pack(side="left")
        ttk.Combobox(row3, textvariable=self.device_var, values=["auto", "cpu", "cuda"], width=10, state="readonly").pack(
            side="left", padx=(8, 14)
        )
        ttk.Label(row3, text="Source lang").pack(side="left")
        ttk.Combobox(row3, textvariable=self.source_lang_var, values=["auto", "eng", "urd"], width=8, state="readonly").pack(
            side="left", padx=(8, 14)
        )
        ttk.Label(row3, text="Target lang").pack(side="left")
        ttk.Entry(row3, textvariable=self.target_lang_var, width=8).pack(side="left", padx=(8, 14))
        ttk.Label(row3, text="Voice gender").pack(side="left")
        ttk.Combobox(
            row3, textvariable=self.voice_gender_var, values=["auto", "male", "female"], width=8, state="readonly"
        ).pack(side="left", padx=(8, 14))

        row4 = ttk.Frame(frm)
        row4.pack(fill="x", pady=4)
        ttk.Checkbutton(row4, text="Enable Translation Verification", variable=self.verify_var).pack(side="left")
        ttk.Label(row4, text="Chunk seconds").pack(side="left")
        ttk.Spinbox(row4, from_=8, to=50, textvariable=self.chunk_var, width=6).pack(side="left", padx=(8, 20))
        ttk.Label(row4, text="Model").pack(side="left")
        ttk.Entry(row4, textvariable=self.model_var, width=42).pack(side="left", padx=8)

        row5 = ttk.Frame(frm)
        row5.pack(fill="x", pady=8)
        ttk.Button(row5, text="Add To Queue", command=self._enqueue_job).pack(side="left")
        ttk.Button(row5, text="Start Queue", command=self._start_worker).pack(side="left", padx=8)
        ttk.Button(row5, text="Stop After Current", command=self._stop_worker).pack(side="left")
        ttk.Button(row5, text="Open Output Folder", command=self._open_output_folder).pack(side="left", padx=8)

        ttk.Separator(frm).pack(fill="x", pady=8)
        ttk.Label(frm, text="Queue").pack(anchor="w")
        self.queue_list = tk.Listbox(frm, height=9)
        self.queue_list.pack(fill="x", pady=4)

        self.progress = ttk.Progressbar(frm, maximum=1.0, variable=self.progress_var)
        self.progress.pack(fill="x", pady=8)
        ttk.Label(frm, textvariable=self.status_var).pack(anchor="w")
        out_row = ttk.Frame(frm)
        out_row.pack(fill="x", pady=(4, 2))
        ttk.Label(out_row, text="Last output file").pack(side="left")
        ttk.Entry(out_row, textvariable=self.last_output_var, width=95).pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(out_row, text="Open File", command=self._open_last_output_file).pack(side="left")

        ttk.Label(frm, text="Logs").pack(anchor="w", pady=(10, 0))
        self.log_box = ScrolledText(frm, height=20, wrap="word")
        self.log_box.pack(fill="both", expand=True, pady=4)
        self.log_box.configure(state="disabled")

    def _log(self, message: str) -> None:
        self.log_queue.put(message)

    def _progress(self, value: float, message: str) -> None:
        self.progress_queue.put((max(0.0, min(1.0, value)), message))

    def _pick_input(self) -> None:
        p = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4;*.mkv;*.mov;*.avi"), ("All files", "*.*")],
        )
        if p:
            self.input_var.set(p)

    def _pick_output(self) -> None:
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_var.set(p)

    def _open_output_folder(self) -> None:
        out_dir = Path(self.output_var.get().strip() or str(OUTPUTS_DIR))
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(out_dir))
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open folder: {exc}")

    def _open_last_output_file(self) -> None:
        p = Path(self.last_output_var.get().strip())
        if not p.exists():
            messagebox.showerror("Missing file", "No completed output file found yet.")
            return
        try:
            import os

            os.startfile(str(p))
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open file: {exc}")

    def _enqueue_job(self) -> None:
        input_path = Path(self.input_var.get().strip())
        if not input_path.exists():
            messagebox.showerror("Missing file", "Select a valid input video first.")
            return
        output_root = Path(self.output_var.get().strip() or str(OUTPUTS_DIR))
        output_root.mkdir(parents=True, exist_ok=True)
        job = UiJob(input_video=input_path, output_root=output_root)
        self.jobs.append(job)
        self._refresh_queue()
        self._log(f"Queued: {input_path}")

    def _refresh_queue(self) -> None:
        self.queue_list.delete(0, tk.END)
        for idx, job in enumerate(self.jobs, start=1):
            suffix = f" -> {job.result_path}" if job.result_path else ""
            err = f" [ERROR: {job.error}]" if job.error else ""
            self.queue_list.insert(tk.END, f"{idx}. {job.input_video.name} [{job.status}]{suffix}{err}")

    def _start_worker(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Queue running", "Queue is already running.")
            return
        if not self.jobs:
            messagebox.showerror("No jobs", "Add at least one job to the queue.")
            return
        self.stop_requested = False
        self.worker_thread = threading.Thread(target=self._worker_main, daemon=True)
        self.worker_thread.start()

    def _stop_worker(self) -> None:
        self.stop_requested = True
        self._log("Stop requested. Current job will finish, then queue stops.")

    def _make_pipeline(self) -> VideoDubPipeline:
        cfg = PipelineConfig(
            model_name=self.model_var.get().strip(),
            source_lang=self.source_lang_var.get().strip() or "auto",
            target_lang=self.target_lang_var.get().strip() or "pbt",
            device_policy=self.device_var.get().strip(),
            chunk_seconds=int(self.chunk_var.get()),
            tts_gender_mode=self.voice_gender_var.get().strip() or "auto",
            enable_translation_verification=bool(self.verify_var.get()),
        )
        return VideoDubPipeline(cfg, log_fn=self._log)

    def _worker_main(self) -> None:
        pipeline = self._make_pipeline()
        for job in self.jobs:
            if self.stop_requested:
                break
            if job.status in {"done", "running"}:
                continue
            try:
                job.status = "running"
                self._refresh_queue()
                self._progress(0.0, f"Running: {job.input_video.name}")
                result = pipeline.run(job.input_video, job.output_root, progress_fn=self._progress)
                job.status = "done"
                job.result_path = result.final_video
                self.last_output_var.set(str(result.final_video))
                self._log(f"Completed: {result.final_video}")
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                self._log(f"Failed {job.input_video.name}: {exc}")
            finally:
                self._refresh_queue()
                self._progress(0.0, "Idle")
        self._log("Queue finished.")

    def _poll_queues(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert(tk.END, message + "\n")
                self.log_box.see(tk.END)
                self.log_box.configure(state="disabled")
        except Empty:
            pass

        try:
            while True:
                value, message = self.progress_queue.get_nowait()
                self.progress_var.set(value)
                self.status_var.set(message)
        except Empty:
            pass

        self.root.after(150, self._poll_queues)


def launch_gui() -> None:
    root = tk.Tk()
    app = DubbingGui(root)
    app._log("GUI ready. Add videos to queue and press 'Start Queue'.")
    root.mainloop()

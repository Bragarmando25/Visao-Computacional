"""Janela do programa para selecionar arquivos e acompanhar o processamento.

Contém Selecionar arquivo, Processar, Parar e Abrir resultados. O processamento
executa em uma tarefa separada para que a janela continue respondendo, enquanto
a prévia e as mensagens são atualizadas pelo thread principal do Tkinter.
"""
from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .vision import load_config, ROOT
from .processing import process_image, process_video, IMAGE_SUFFIXES


def launch():
    # Alguns Python portáteis não localizam o Tcl em diretórios de aplicativos.
    # A instalação normal do Python já traz essa biblioteca no lugar correto.
    portable_tcl = ROOT / ".venv" / "tcl"
    if (portable_tcl / "tcl8.6" / "init.tcl").exists():
        os.environ.setdefault("TCL_LIBRARY", str(portable_tcl / "tcl8.6"))
        os.environ.setdefault("TK_LIBRARY", str(portable_tcl / "tk8.6"))
    root = tk.Tk()
    root.title("Doces e ferramentas | Visão Computacional")
    root.geometry("1260x850")
    root.minsize(950, 670)
    root.configure(bg="#10151f")
    messages = queue.Queue()
    running, stop_event = False, threading.Event()
    selected = []
    image_ref = None
    current_preview = None
    destination = ROOT / "resultados"
    config = load_config()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TButton", font=("Segoe UI", 11), padding=(14, 10))
    style.configure("Accent.TButton", background="#8463ed", foreground="white")
    style.map("Accent.TButton", background=[("active", "#9d82ff")])
    style.configure("TProgressbar", troughcolor="#263044", background="#a18aff")

    top = tk.Frame(root, bg="#10151f")
    top.pack(fill="x", padx=26, pady=(22, 15))
    tk.Label(top, text="Doces + ferramentas", font=("Segoe UI", 24, "bold"), fg="#f5f5fc", bg="#10151f").pack(anchor="w")
    tk.Label(top, text="Contagem, cores e medidas · Fotos e vídeos", font=("Segoe UI", 11), fg="#a8b5ca", bg="#10151f").pack(anchor="w", pady=(5, 0))
    bar = tk.Frame(root, bg="#10151f")
    bar.pack(fill="x", padx=26)
    status = tk.StringVar(value="Selecione uma foto ou um vídeo para começar.")
    selection = tk.StringVar(value="Nenhum arquivo selecionado")

    def choose():
        nonlocal selected
        paths = filedialog.askopenfilenames(title="Selecionar fotos ou vídeos", initialdir=ROOT,
                    filetypes=[("Fotos e vídeos", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv")])
        if paths:
            selected = [Path(p) for p in paths]
            selection.set(selected[0].name if len(selected) == 1 else f"{len(selected)} arquivos selecionados")

    def worker(paths):
        try:
            for path in paths:
                if stop_event.is_set():
                    break
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    process_image(path, destination, config, progress=update)
                else:
                    process_video(path, destination, config, progress=update, stop=stop_event.is_set)
            messages.put(("done", "Resultados salvos na pasta resultados."))
        except Exception as error:
            messages.put(("error", str(error)))

    def update(message, image):
        # Só o thread principal toca nos elementos do Tk.
        messages.put(("update", (message, image)))

    def start():
        nonlocal running
        if running:
            return
        if not selected:
            messagebox.showinfo("Selecionar arquivo", "Selecione ao menos uma foto ou um vídeo.")
            return
        running = True
        stop_event.clear()
        status.set("Processando…")
        progress.start(12)
        run_button.state(["disabled"])
        threading.Thread(target=worker, args=(selected.copy(),), daemon=True).start()

    ttk.Button(bar, text="Selecionar arquivos", command=choose).pack(side="left", padx=(0, 8))
    run_button = ttk.Button(bar, text="Processar", style="Accent.TButton", command=start)
    run_button.pack(side="left", padx=(0, 8))
    ttk.Button(bar, text="Parar", command=stop_event.set).pack(side="left", padx=(0, 8))
    ttk.Button(bar, text="Abrir resultados", command=lambda: (destination.mkdir(exist_ok=True), os.startfile(destination))).pack(side="right")
    tk.Label(root, textvariable=selection, font=("Segoe UI", 10), fg="#c3cce0", bg="#10151f", anchor="w").pack(fill="x", padx=28, pady=(10, 7))
    canvas = tk.Canvas(root, bg="#181f2c", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=26)
    canvas.create_text(400, 200, text="O resultado aparecerá aqui", fill="#8f9db5", font=("Segoe UI", 18))
    footer = tk.Frame(root, bg="#10151f")
    footer.pack(fill="x", padx=26, pady=(10, 18))
    tk.Label(footer, textvariable=status, font=("Segoe UI", 10), fg="#c3cce0", bg="#10151f", anchor="w").pack(fill="x")
    progress = ttk.Progressbar(footer, mode="indeterminate")
    progress.pack(fill="x", pady=(8, 0))

    def draw_preview():
        nonlocal image_ref
        if current_preview is None:
            return
        im = Image.fromarray(cv2.cvtColor(current_preview, cv2.COLOR_BGR2RGB))
        im.thumbnail((max(100, canvas.winfo_width() - 16), max(100, canvas.winfo_height() - 16)))
        image_ref = ImageTk.PhotoImage(im)
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=image_ref)

    def poll():
        nonlocal running, current_preview
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == "update":
                    status.set(value[0])
                    current_preview = value[1]
                    draw_preview()
                elif kind in ("done", "error"):
                    running = False
                    progress.stop()
                    run_button.state(["!disabled"])
                    status.set(value)
                    if kind == "error":
                        messagebox.showerror("Falha no processamento", value)
        except queue.Empty:
            pass
        root.after(100, poll)

    def close():
        if running:
            stop_event.set()
            status.set("Encerrando e salvando o vídeo…")
            root.after(150, close_when_done)
        else:
            root.destroy()

    def close_when_done():
        if running:
            root.after(150, close_when_done)
        else:
            root.destroy()

    canvas.bind("<Configure>", lambda _: draw_preview())
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, poll)
    root.mainloop()

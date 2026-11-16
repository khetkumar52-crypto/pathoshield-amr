# PathoShield AI 🧬
*Fast genomic scanning for superbugs.*

## The Problem
Here in Sindh, Pakistan, we have a massive crisis with superbugs like XDR Typhoid. Right now, hospitals wait 2 to 3 days to grow bacteria in a petri dish just to figure out which antibiotics will work. During that 72-hour wait, patients are given the wrong medicines and the infection gets worse.

## What This Is
I am a 14-year-old student, and I built PathoShield AI to skip the petri dish. It acts like a DNA barcode scanner. You feed it the raw genetic letters (A, T, C, G) of a bacterium, and it instantly scans for "armor" genes—the exact mutations that make the bug immune to medicine.

It detects 10 high-risk resistance markers (like `blaCTX-M-15` and `blaNDM-1`) in under 2 seconds.

## Links
- **Live Web App:** [Hugging Face Space](https://huggingface.co/spaces/khetkumar/pathoshield-amr) *(Note: If the space says "Sleeping", just wait 30 seconds for it to wake up).*
- **Technical Overview:** [Google Doc](https://docs.google.com/document/d/12X4Hd-o8WRdcWidK19a5kG-9oOx0XAs2wWcuh5N5Oag/edit)
- **Validation Database:** [Google Sheet](https://docs.google.com/spreadsheets/d/1-a9_17FQeIZ0TBTuJYGtMJzAuI3En7NKdsRJdygYOXY/edit)

## How It Works (The Tech)
- **Language:** Python
- **Interface:** Gradio
- **Machine Learning:** Scikit-learn (`RandomForestClassifier`)
- **Input:** Raw nucleotide strings or `.fasta` files.

## Builder's Honesty Note
I built this as a high school proof-of-concept to prove that software can speed up diagnostic triage. Right now, the engine achieves 100% accuracy on a benchmark of 15 real clinical isolates from the NCBI database. 

However, to keep it lightning-fast for the web demo, the current prototype uses exact Python string-matching (rather than complex bioinformatics alignment) and trains on a synthetic dataset of 2,500 mock genomes when the app boots up. The ultimate goal is to upgrade this logic and validate it against large-scale clinical hospital datasets.

## How to Run It Locally
If you want to run the code on your own computer:
1. Clone this repository: `git clone https://github.com/khetkumar52-crypto/pathoshield-amr.git`
2. Install the required libraries (you will need `gradio`, `pandas`, `numpy`, and `scikit-learn`).
3. Run the app: `python app.py`
4. Open the local web link it gives you in your browser.

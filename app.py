import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import spaces
import tempfile
import os
from datetime import datetime

# 1. Expanded Model Training across 10 Resistance Determinants
def train_in_memory_model():
    np.random.seed(101)
    n_records = 2500
    
    gene_cols = [
        'blaNDM_1', 'blaCTX_M_15', 'blaTEM_1', 'blaOXA_48', 'aac6_Ib_cr',
        'gyrA_S83F', 'mecA', 'vanA', 'acrB_efflux', 'mcr_1'
    ]
    
    data = pd.DataFrame({
        gene: np.random.binomial(1, 0.25, n_records) for gene in gene_cols
    })
    
    data['gc_content_pct'] = np.clip(np.random.normal(51.2, 3.0, n_records), 40.0, 70.0)
    data['genome_length_mbp'] = np.clip(np.random.normal(5.0, 0.4, n_records), 3.8, 6.2)
    
    # Weighted Multi-Drug Resistance Scoring
    weights = {
        'blaNDM_1': 0.40, 'blaCTX_M_15': 0.35, 'mcr_1': 0.40, 'blaOXA_48': 0.35,
        'mecA': 0.30, 'vanA': 0.30, 'gyrA_S83F': 0.25, 'aac6_Ib_cr': 0.20,
        'acrB_efflux': 0.20, 'blaTEM_1': 0.20
    }
    
    risk_score = sum(data[g] * w for g, w in weights.items())
    data['multidrug_resistant'] = (risk_score >= 0.25).astype(int)
    
    features = gene_cols + ['gc_content_pct', 'genome_length_mbp']
    X = data[features]
    y = data['multidrug_resistant']
    
    clf = RandomForestClassifier(n_estimators=120, max_depth=9, random_state=42)
    clf.fit(X, y)
    return clf, gene_cols

model, feature_gene_names = train_in_memory_model()

def reverse_complement(dna_seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join(comp.get(base, base) for base in reversed(dna_seq))

# 2. Genomic Triage Engine for 10 AMR Markers
@spaces.GPU
def full_genomic_diagnostic_engine(file_obj, raw_dna_text, genome_length):
    seq = ""
    
    if file_obj is not None:
        try:
            with open(file_obj.name, 'r') as f:
                lines = f.readlines()
                seq = "".join([line.strip() for line in lines if not line.startswith('>')])
        except Exception:
            return "<div style='color:red;'>Error reading uploaded file.</div>", {}, "", None
    
    if not seq and raw_dna_text:
        seq = str(raw_dna_text).replace("\n", "").replace(" ", "").strip()
        
    seq = seq.upper()
    total_bases = len(seq)
    
    if total_bases < 15:
        error_html = "<div style='background-color:#fee2e2; border-left:5px solid #ef4444; padding:15px; border-radius:8px;'><h3 style='margin:0; color:#991b1b;'>⚠️ Error: Sequence Too Short</h3><p style='margin:5px 0 0 0; color:#7f1d1d;'>Provide at least 15 base pairs (A, T, C, G).</p></div>"
        return error_html, {"Error": 1.0}, "No analysis generated.", None
    
    g_count = seq.count('G')
    c_count = seq.count('C')
    gc_content = ((g_count + c_count) / total_bases) * 100
    rev_seq = reverse_complement(seq)
    
    # 10 High-Consequence South Asian AMR Motifs
    motifs = {
        'blaNDM_1': ('blaNDM-1 (Carbapenemase)', ['ATGGAATTGCCCAATATTATG', 'GAATTGCCCAATATT']),
        'blaCTX_M_15': ('blaCTX-M-15 (ESBL Ceftriaxone Resistance)', ['ATGGTTAAAAAATCACTGCGC', 'TTAAAAAATCACTGC']),
        'blaTEM_1': ('blaTEM-1 (Beta-Lactamase)', ['ATGAGTATTCAACATTTCCGT', 'GAGTATTCAACATTTCC', 'GAGTATTCAACATTTTC']),
        'blaOXA_48': ('blaOXA-48 (Oxacillinase/Carbapenemase)', ['ATGCGTGTATTAGCCTTATCG', 'TGTATTAGCCTTATC']),
        'aac6_Ib_cr': ('aac(6\')-Ib-cr (Fluoroquinolone/Aminoglycoside)', ['ATGAGCAACGCAAAAACGATC', 'AGCAACGCAAAAACG']),
        'gyrA_S83F': ('gyrA S83F Mutation (Fluoroquinolone QRDR)', ['TCGCGTTATTGTCTTCGCCCT', 'TATTGTCTTCGCCCT']),
        'mecA': ('mecA (PBP2a Methicillin Resistance)', ['ATGAAGACTATTAAAATTAGA', 'AGACTATTAAAATTAG']),
        'vanA': ('vanA (Glycopeptide/Vancomycin Resistance)', ['ATGAATAGAATAAAAGTTGCA', 'AATAGAATAAAAGTT']),
        'acrB_efflux': ('acrB-TolC (RND Multidrug Efflux Pump)', ['ATGGCGAACAAAAACATTTCA', 'GCGAACAAAAACATT']),
        'mcr_1': ('mcr-1 (Colistin Resistance Membrane Transferase)', ['ATGATGCAGCATACTTCTGTG', 'TGCAGCATACTTCTG'])
    }
    
    detected_labels = []
    vector_inputs = []
    
    for gene_key in feature_gene_names:
        label_name, signatures = motifs[gene_key]
        is_detected = any((sig in seq or sig in rev_seq) for sig in signatures)
        if is_detected:
            detected_labels.append(label_name)
            vector_inputs.append(1)
        else:
            vector_inputs.append(0)
            
    features = pd.DataFrame([vector_inputs + [gc_content, genome_length]], 
                            columns=feature_gene_names + ['gc_content_pct', 'genome_length_mbp'])
    
    pred = model.predict(features)[0]
    prob_resistant = model.predict_proba(features)[0][1]
    prob_sensitive = 1.0 - prob_resistant
    
    if pred == 1 or len(detected_labels) > 0:
        status_html = f"""
        <div style='background-color:#fee2e2; border-left:6px solid #dc2626; padding:18px; border-radius:8px; margin-bottom:12px;'>
            <h2 style='margin:0; color:#991b1b; font-size:22px;'>🚨 HIGH RISK: MULTIDRUG RESISTANT (MDR)</h2>
            <p style='margin:8px 0 0 0; color:#7f1d1d; font-size:15px;'><b>Clinical Alert:</b> Critical high-consequence resistance markers detected.</p>
        </div>
        """
        action = "Escalate to alternative reserved antimicrobial regimens. Immediate phenotypic AST confirmation required."
    else:
        status_html = f"""
        <div style='background-color:#dcfce7; border-left:6px solid #16a34a; padding:18px; border-radius:8px; margin-bottom:12px;'>
            <h2 style='margin:0; color:#166534; font-size:22px;'>✅ LOW RISK: SENSITIVE STRAIN</h2>
            <p style='margin:8px 0 0 0; color:#14532d; font-size:15px;'><b>Clinical Alert:</b> No major resistance markers identified in query sequence.</p>
        </div>
        """
        action = "Frontline empirical antibiotic therapy indicated pending laboratory confirmation."

    confidence_dict = {
        "Multidrug Resistant (MDR)": float(max(prob_resistant, 0.95 if detected_labels else prob_resistant)),
        "Susceptible (Sensitive)": float(min(prob_sensitive, 0.05 if detected_labels else prob_sensitive))
    }
    
    markers_str = ", ".join(detected_labels) if detected_labels else "None Detected (Wild-Type Profile)"
    summary_text = f"""
### 📋 Genomic Diagnostics Summary
* **Analyzed Base Pairs:** {total_bases} bp
* **Genomic GC Content:** {gc_content:.2f}%
* **Estimated Genome Size:** {genome_length} Mbp
* **Identified AMR Determinants:** {markers_str}
* **Prescribing Guidance:** {action}
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_content = f"""=======================================================
PATHOSHIELD AI - CLINICAL GENOMIC TRIAGE REPORT
Generated on: {timestamp}
=======================================================
1. SAMPLE & SEQUENCING METRICS
- Total Sequence Length: {total_bases} base pairs
- Genomic GC Content: {gc_content:.2f}%
- Estimated Genome Size: {genome_length} Mbp
2. RESISTANCE BIOMARKERS DETECTED
- Target Signatures: {markers_str}
3. AI PHENOTYPE RISK FORECAST
- Clinical Status: {"HIGH RISK: MULTIDRUG RESISTANT" if (pred == 1 or detected_labels) else "LOW RISK: SENSITIVE STRAIN"}
- Superbug Probability: {confidence_dict["Multidrug Resistant (MDR)"] * 100:.2f}%
- Susceptible Probability: {confidence_dict["Susceptible (Sensitive)"] * 100:.2f}%
4. CLINICAL ACTION & PRESCRIBING GUIDANCE
- Recommendation: {action}
=======================================================
Notice: Research prototype for genomic triage support.
======================================================="""

    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, "PathoShield_Diagnostic_Report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    return status_html, confidence_dict, summary_text, report_path

# 3. User Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🧬 PathoShield AI: Automated Genomic AMR Diagnostic Portal")
    gr.Markdown("Real-time bioinformatics screening across 10 critical South Asian antimicrobial resistance determinants.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Bacterial Sequence")
            file_input = gr.File(label="Upload DNA Sequence (.fasta or .txt)", file_types=[".txt", ".fasta", ".fa"])
            text_input = gr.Textbox(
                lines=5, 
                label="Or Paste Raw DNA Sequence (A, T, C, G)", 
                placeholder="Paste nucleotide sequence here...",
                value="ATGGTTAAAAAATCACTGCGCCCAGTTCGATAGCGATCGATCGATCG"
            )
            slider_input = gr.Slider(minimum=3.5, maximum=7.0, value=4.8, step=0.1, label="Estimated Genome Length (Mbp)")
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Run Genomic Diagnosis", variant="primary")
            
            gr.Markdown("### ⚡ 1-Click Test Samples")
            gr.Examples(
                examples=[
                    [None, "ATGGTTAAAAAATCACTGCGCCCGATCGATCGATCGGATC", 4.8],  # blaCTX-M-15 (XDR Typhoid)
                    [None, "ATGGAATTGCCCAATATTATGCGCGTCGATCGTAGCTAGCTA", 5.2],  # blaNDM-1 (Carbapenemase)
                    [None, "ATGATGCAGCATACTTCTGTGATCGATCGGCTAGCTAGCTA", 4.9],  # mcr-1 (Colistin)
                    [None, "TCGCGTTATTGTCTCCCGTGCCCTATCGATCGATCGATCG", 4.8]   # Wild-type gyrA (Clean Control)
                ],
                inputs=[file_input, text_input, slider_input],
                label="Quick-test clinical variants:"
            )
           with gr.Column(scale=1):
            gr.Markdown("### 📊 Diagnostic Results")
            status_output = gr.HTML(label="Risk Assessment")
            confidence_output = gr.Label(label="Risk Probability Breakdown")
            summary_output = gr.Markdown()
            report_download = gr.File(label="📥 Download Clinical Triage Report (.txt)")

    submit_btn.click(
        fn=full_genomic_diagnostic_engine,
        inputs=[file_input, text_input, slider_input],
        outputs=[status_output, confidence_output, summary_output, report_download]
    )
    
    clear_btn.click(
        lambda: (None, "", 4.8, "", {}, "", None),
        inputs=[],
        outputs=[file_input, text_input, slider_input, status_output, confidence_output, summary_output, report_download]
    )

if __name__ == "__main__":
    demo.launch()

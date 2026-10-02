#!/usr/bin/env python3
"""
Wrapper de compatibilidade para gerar e pré-renderizar o notebook de análise comparativa:
Analise_Comparativa_Metricas_Esqueletizacao.ipynb
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(__file__), "generate_and_execute_notebook.py")
    subprocess.run([sys.executable, script_path], check=True)

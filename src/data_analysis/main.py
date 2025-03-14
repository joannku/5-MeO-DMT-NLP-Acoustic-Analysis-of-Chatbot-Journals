"""
This python file can be run to execute all analyses in the correct order. 
"""

import subprocess

subprocess.run(['python', '1_summary_statistics.py'])
subprocess.run(['python', '2_mixed_models_liwc.py'])
subprocess.run(['python', '3_mixed_models.goemo.py'])
subprocess.run(['python', '4_vocal_prepost.py'])
subprocess.run(['python', '5_pca_goemo.py'])
subprocess.run(['python', '6_pca_goemo+liwc.py'])
subprocess.run(['python', '7_vocal_dynamics.py'])

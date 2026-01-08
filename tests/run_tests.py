#!/usr/bin/env python3
"""Simple test runner"""
import os
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.exit(subprocess.run([sys.executable, '-m', 'pytest'] + sys.argv[1:]).returncode)

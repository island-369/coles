#!/usr/bin/env python3

print("Testing debug_config import...")

try:
    from debug_config import LOG_TO_FILE, LOG_FILE_PATH
    print(f"LOG_TO_FILE: {LOG_TO_FILE}")
    print(f"LOG_FILE_PATH: {LOG_FILE_PATH}")
except ImportError as e:
    print(f"Import error: {e}")

print("Config test completed")
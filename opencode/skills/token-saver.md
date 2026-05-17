---
name: token-saver
description: Context compression & efficient prompting patterns for RDNA2 work
triggers: ["token", "context", "compress", "summarize", "truncate", "efficient"]
---
# Token-Saving Patterns for RDNA2 Development
### 5. `token-saver.md` *(Critical for Token Efficiency)*
## 1. Context Compression Rules
✅ **Do**: Reference files by path + line range  
   `"See mmvq.cu:142-158 for stream-k logic"`  
❌ **Don't**: Paste entire kernel → wastes 200+ tokens  

✅ **Do**: Use ISA mnemonics, not full encoding  
   `"Use global_load_dword ... slc"`  
❌ **Don't**: Include full 64-bit instruction encoding  

✅ **Do**: Summarize telemetry deltas  
   `"WAVE_ISSUE_WAIT ↓18%, VGPR=38"`  
❌ **Don't**: Paste full rocprofv3 SQLite dump  

## 2. Prompt Templates (Token-Efficient)
### Code Review Request

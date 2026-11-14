---
name: corpus-fix-long-katakana
description: Review and correct long continuous katakana in the shared flat-text corpus according to the canonical segmentation policy.
metadata:
  short-description: Follow the long-katakana segmentation policy
---

# Long katakana corpus correction

This skill is an entry point, not a separate source of segmentation rules. Before doing any long-katakana work, read and follow the authoritative policy at `D:\Git\corpus-management\docs\katakana-split.md`. It governs candidate screening, main/SubAI responsibilities, Sudachi, context and evidence review, POS, canonical edits, reporting, and schedule-task behavior. Do not follow conflicting historical instructions or continuation notes from this skill file.

Use the policy document's linked corpus, dictionary-build, and POS references for their respective procedures. Keep run-specific state in the currently authorized work record. A user stop or pause remains in force until the user explicitly resumes the work; do not restart processing or recreate its schedule task while it is stopped.

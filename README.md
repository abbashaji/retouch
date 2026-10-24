# Marquise -> Pear retouch (Windows, no GPU)

Install once (Python 3.9+):   pip install opencv-python numpy

## Recommended: TILE workflow (much sharper - each stone gets ~300px in Gemini)

For each photo (script + photo in one folder):

1. python pear_pipeline.py detect photo.jpg
   Click window: left-click = add stone, right-click = remove, Enter when 15/15.
2. python pear_pipeline.py prepare-tiles photo.jpg
   Makes photo_tile1_for_gemini.png ... photo_tile5_for_gemini.png + matching *_prompt.txt.
3. For EACH tile: upload it to Gemini with its prompt text, generate 4-6 times,
   save the results as photo_tile1_out1.jpg, photo_tile1_out2.jpg, ... (tile number = k).
4. python pear_pipeline.py composite-tiles photo.jpg
   Matches stones by position, picks the best version of each stone across ALL outputs,
   pastes them onto the ORIGINAL full-resolution photo. Writes photo_final_tiles.png,
   photo_final_tiles_compare.jpg and a per-stone report. Stones marked "<-- check" or
   "NO usable candidate" need another generation or hand retouching.

Test first with ONE photo and look at the result before doing all five.

## Old whole-photo workflow (lower quality): prepare / composite

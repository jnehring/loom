# Records docs/demo.gif: a real OpenAI batch run with loom-batch.
# Run from the repo root with OPENAI_API_KEY set:  vhs demo/demo.tape
Output docs/demo.gif

Set Shell bash
Set FontSize 18
Set Width 1300
Set Height 600
Set Padding 20
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"

Hide
Type "export LOOM_HOME=$(mktemp -d) PATH=$PWD/.venv/bin:$PATH DEMO=$PWD/demo && cd $(mktemp -d) && cp $DEMO/reviews.csv . && export PS1='$ ' && clear"
Enter
Sleep 1s
Show

Type "cat reviews.csv"
Enter
Sleep 2s

Type `loom run -f reviews.csv -p openai -m gpt-5.4-mini --system "Classify the sentiment as positive, negative, or neutral. Answer with one word."`
Enter
Wait+Screen /Batch submitted/
Sleep 2s

Type "loom fetch"
Enter
Sleep 4s

# Cut the wait for the batch to finish out of the recording.
Hide
Type "python $DEMO/wait_for_batch.py && echo BATCH_$((1+1))"
Enter
Wait+Screen@180m /BATCH_2/
Type "clear"
Enter
Sleep 1s
Show

Type "# ...later"
Sleep 1s
Enter
Type "loom fetch"
Enter
Wait+Screen /complete/
Sleep 2s

Type "cat reviews_results_openai_gpt-5.4-mini.csv"
Enter
Sleep 5s

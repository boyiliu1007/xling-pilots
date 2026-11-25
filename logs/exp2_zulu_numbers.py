exec(open("logs/exp2_copsd_setup_numbers.py").read().split('print("== Zulu")')[0])
L="results/exp2_qwen3-4b/langeval/"
Z={"J":{42:load(L+"grpo__gsm8k-mt3k-zulonly__exact__votelidrefpw-lr5e-6-q16-steps300__s42__zul/eval/step300_test.jsonl","zul")},
   "D":{42:load(L+"grpo__gsm8k-mt3k-zulonly__exact__votelidref-lr5e-6-q16-steps300__s42__zul/eval/step300_test.jsonl","zul")},
   "G":{42:load(L+"grpo__gsm8k-mt3k-zulonly__exact__voteonly-lr5e-6-q16-steps300__s42__zul/eval/step300_test.jsonl","zul")},
   "base":{42:load(L+"base__zul/eval/base_test.jsonl","zul")}}
for k,v in Z.items(): rep(k,v)
comp("J","D",Z["J"],Z["D"]); comp("J","base",Z["J"],Z["base"]); comp("D","base",Z["D"],Z["base"])

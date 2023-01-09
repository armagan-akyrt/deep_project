from transformers import pipeline, MarianMTModel, MarianTokenizer
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_load_path', default=os.getcwd(), type=str)
    parser.add_argument('--input_file', default="input.txt", type=str)
    parser.add_argument('--output_file', default="output.txt", type=str)
    parser.add_argument('--checkpoint', default=1000, type=str)
    
    args = vars(parser.parse_args())
    
    inp = open(args['input_file'], "r")
    model_path = args['model_load_path'] + "/opus-mt-tr-en-finetuned-tr-to-en/checkpoint-" + args['checkpoint']
    model = MarianMTModel.from_pretrained(model_path)   # loads the model
    tokenizer = MarianTokenizer.from_pretrained(model_path) # tokenizes the model




    classifier = pipeline('translation_tr_to_en', model=model, tokenizer=tokenizer)
    inp_arr  = []
    for line in inp.readlines():
        if len(line) > 1:
            inp_arr.append(line)
        
    print(inp_arr)
    res = classifier(inp_arr)

    out = open(args['output_file'], "w")
    out.truncate(0)   # clear prev. outpur

    for i in range(0, len(res)):
        
        out.writelines(res[i]['translation_text'] + '\n')


    print(res)

    # close flies.
    inp.close()
    out.close()
    
main()

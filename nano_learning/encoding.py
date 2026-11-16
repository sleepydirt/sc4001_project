"""Lossless retained-subword fallback; never lower-case argument values."""
def retained_encode(tokenizer,inverse,text):
    ids=tokenizer.encode(text,add_special_tokens=False)
    if all(i in inverse for i in ids):return [inverse[i] for i in ids]
    # Recursively split only unsupported spans. Character-level byte pieces for
    # the lowercase ASCII JSON surface remain in the released vocabulary.
    if len(text)<=1:raise ValueError('Current retained-token interface cannot encode character')
    middle=len(text)//2
    result=retained_encode(tokenizer,inverse,text[:middle])+retained_encode(tokenizer,inverse,text[middle:])
    mapping={v:k for k,v in inverse.items()}
    decoded=tokenizer.decode([mapping[i] for i in result],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    if decoded!=text:raise ValueError('Retained-token fallback changed text')
    return result

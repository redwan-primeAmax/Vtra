import re
from typing import List, Dict
from deepmultilingualpunctuation import PunctuationModel
from dubbing.memory import flush_memory

def restore_punctuation_and_sentences(segments: List[Dict]) -> List[Dict]:
    """
    ১. DeepMultilingualPunctuation AI মডেল দিয়ে নিখুঁত পাংচুয়েশন বসানো।
    ২. টাইমস্ট্যাম্প ও পাংচুয়েশন মিলিয়ে বাক্য আলাদা করা।
    """
    # ১. সমস্ত ওয়ার্ড এবং টাইমস্ট্যাম্পের একটি ফ্ল্যাট লিস্ট তৈরি
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        return segments

    # ২. পুরো টেক্সট একত্রিত করা AI মডেল দিয়ে পাংচুয়েট করার জন্য
    raw_text = " ".join([w["word"].strip() for w in all_words])
    
    # NLP Punctuation Model লোড (CPU-তেই দ্রুত চলে)
    model = PunctuationModel()
    punctuated_text = model.restore_punctuation(raw_text)
    
    # মডেলের কাজ শেষ, মেমোরি ক্লিয়ার
    del model
    flush_memory()

    # ৩. পাংচুয়েটেড টেক্সটের শব্দগুলোকে আলাদা করে অরিজিনাল টাইমস্ট্যাম্পের সাথে ম্যাপিং
    punct_words = punctuated_text.split()
    
    sentences = []
    current_sentence_words = []
    current_start = None

    for i, word_info in enumerate(all_words):
        if current_start is None:
            current_start = word_info["start"]
            
        # এআই প্রদত্ত পাংচুয়েটেড ওয়ার্ড ব্যবহার
        punct_word = punct_words[i] if i < len(punct_words) else word_info["word"]
        current_sentence_words.append(punct_word)

        is_last = (i == len(all_words) - 1)
        
        # বাক্যের শেষে দাড়ি/প্রশ্নবোধক/আশ্চর্যবোধক চিহ্ন আছে কি না বা দীর্ঘ বিরতি আছে কি না পরীক্ষা
        ends_with_sentence_punc = bool(re.search(r'[.!?]$', punct_word))
        
        gap = 0 if is_last else (all_words[i+1]["start"] - word_info["end"])

        # বাক্য শেষ করার শর্ত: AI প্রদত্ত পূর্ণচ্ছেদ অথবা ০.৭ সেকেন্ডের বেশি বিরতি
        if ends_with_sentence_punc or gap > 0.7 or is_last:
            sentence_text = " ".join(current_sentence_words).strip()
            
            # ফিলার শব্দ (uh, um, ah) পরিষ্কার করা
            sentence_text = re.sub(r'\b(uh|um|ah)\b', '', sentence_text, flags=re.IGNORECASE)
            sentence_text = re.sub(r'\s+', ' ', sentence_text).strip()

            if sentence_text:
                sentences.append({
                    "start": current_start,
                    "end": word_info["end"],
                    "text": sentence_text
                })
            
            current_sentence_words = []
            current_start = None

    return sentences

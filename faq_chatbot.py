"""
FAQ Chatbot (NLP + Cosine Similarity)
-------------------------------------
Topic: an online learning platform ("EduLearn").

Pipeline
  1. Collect FAQs        -> FAQS list of (question, answer)
  2. Preprocess text     -> lowercase, remove punctuation, tokenize (NLTK),
                            remove stop words, stem (NLTK Porter stemmer)
  3. Match questions     -> TF-IDF vectors + cosine similarity
  4. Display best answer -> shown with a confidence score; a fallback message
                            is used when nothing is similar enough
  5. Optional chat UI    -> simple Tkinter window

Requires: nltk, scikit-learn      (pip install nltk scikit-learn)
Run:      python faq_chatbot.py
"""

import re

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. COLLECT FAQs  (replace these with your own topic/product if you like)
# ---------------------------------------------------------------------------
FAQS = [
    ("How do I create an account?",
     "Click 'Sign Up' on the top right, enter your name, email and a password, "
     "then verify your email address."),
    ("I forgot my password. How can I reset it?",
     "Click 'Forgot Password' on the login page and enter your email. "
     "We will send you a link to set a new password."),
    ("How do I enroll in a course?",
     "Open the course page and click 'Enroll Now'. Free courses start "
     "immediately; paid courses ask you to complete payment first."),
    ("Are there any free courses?",
     "Yes! We offer many free courses. Look for the 'Free' badge on the "
     "course card or use the 'Free' filter on the catalog page."),
    ("What payment methods do you accept?",
     "We accept credit cards, debit cards, UPI, net banking and popular wallets."),
    ("Can I get a refund or my money back?",
     "Yes. You can request a full refund within 7 days of purchase if you have "
     "completed less than 20% of the course."),
    ("Will I get a certificate after finishing a course?",
     "Yes. Once you complete all lessons and pass the final quiz, you can "
     "download a certificate of completion from your dashboard."),
    ("Is the certificate valid and shareable on LinkedIn?",
     "Every certificate has a unique verification link, so you can add it to "
     "your LinkedIn profile or resume."),
    ("How long do I have access to a course?",
     "Once you enroll you get lifetime access to the course, including all "
     "future updates."),
    ("Can I download the videos to watch offline?",
     "Yes, in the mobile app you can download lessons and watch them offline. "
     "Downloads are not available on the website."),
    ("Is there a mobile app?",
     "Yes, the EduLearn app is available for both Android and iOS."),
    ("How can I contact customer support or talk to a human?",
     "Email us at support@edulearn.example or use the live chat on the Help "
     "page. Support is available from 9 AM to 9 PM, Monday to Saturday."),
    ("Do you offer discounts for students?",
     "Yes. Verify your student email or ID under Account Settings to get 30% "
     "off on paid courses."),
    ("How do I cancel my subscription?",
     "Go to Account Settings, choose 'Subscription' and click 'Cancel'. "
     "You will keep access until the end of the billing period."),
    ("Can I change my registered email address?",
     "Yes. Open Account Settings, click 'Edit Email', and confirm the change "
     "using the link we send to your new address."),
    ("What if a video is not playing?",
     "Check your internet connection, refresh the page, or try another "
     "browser. If it still fails, contact support with a screenshot."),
    ("Do I need any prior knowledge to start?",
     "Each course lists its prerequisites on the course page. Many are "
     "beginner-friendly and need no prior knowledge."),
    ("Can I talk to the instructor or ask doubts?",
     "Yes. Every course has a Q&A section where instructors and other "
     "learners answer your doubts, usually within 24 hours."),
    ("How do I delete my account?",
     "Go to Account Settings, choose 'Delete Account' and confirm. Note that "
     "this permanently removes your progress and certificates."),
    ("Do you provide job placement help?",
     "We offer resume reviews, mock interviews and a job board for learners "
     "who complete our career-track programs."),
]

GREETINGS = {"hi", "hello", "hey", "hola", "namaste"}
EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}
FALLBACK = ("Sorry, I couldn't find a good answer to that. Try rephrasing, "
            "or contact support@edulearn.example.")
THRESHOLD = 0.20     # minimum cosine similarity to accept a match


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
_tokenizer = RegexpTokenizer(r"[a-z0-9]+")       # tokenize + drop punctuation
_stemmer = PorterStemmer()
_STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"not", "no"}   # keep negations


def preprocess(text: str) -> str:
    """lowercase -> tokenize -> remove stop words -> stem -> re-join."""
    tokens = _tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    return " ".join(_stemmer.stem(t) for t in tokens)


# ---------------------------------------------------------------------------
# 3. MATCHING
# ---------------------------------------------------------------------------
class FAQBot:
    def __init__(self, faqs=FAQS, threshold=THRESHOLD):
        self.faqs = faqs
        self.threshold = threshold
        # Index the question AND its answer so keywords in either can match
        corpus = [preprocess(q + " " + q + " " + a) for q, a in faqs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def best_matches(self, query, n=3):
        """Return the top-n (score, question, answer) matches."""
        vec = self.vectorizer.transform([preprocess(query)])
        scores = cosine_similarity(vec, self.matrix).ravel()
        top = scores.argsort()[::-1][:n]
        return [(float(scores[i]), *self.faqs[i]) for i in top]

    def reply(self, user_input):
        """Return (answer text, matched question or None, score)."""
        text = user_input.strip()
        words = set(re.findall(r"[a-z]+", text.lower()))

        if not text:
            return "Please type a question.", None, 0.0
        if words & EXIT_WORDS:
            return "Goodbye! Happy learning.", None, 1.0
        if words & GREETINGS and len(words) <= 2:
            return "Hello! Ask me anything about EduLearn.", None, 1.0

        score, question, answer = self.best_matches(text, n=1)[0]
        if score < self.threshold:
            return FALLBACK, None, score
        return answer, question, score


# ---------------------------------------------------------------------------
# 4. CONSOLE CHAT
# ---------------------------------------------------------------------------
def console_chat():
    bot = FAQBot()
    print("EduLearn FAQ Bot: ask me a question (type 'bye' to exit).\n")
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        answer, matched, score = bot.reply(user)
        print(f"Bot: {answer}")
        if matched:
            print(f"     (matched FAQ: \"{matched}\", confidence {score:.0%})")
        print()
        if answer.startswith("Goodbye"):
            break


# ---------------------------------------------------------------------------
# 5. OPTIONAL: TKINTER CHAT UI
# ---------------------------------------------------------------------------
def gui_chat():
    import tkinter as tk
    from tkinter import scrolledtext

    bot = FAQBot()
    root = tk.Tk()
    root.title("EduLearn FAQ Chatbot")
    root.geometry("480x560")

    chat = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled",
                                     font=("Segoe UI", 10), padx=8, pady=8)
    chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
    chat.tag_config("user", foreground="#0b5394", font=("Segoe UI", 10, "bold"))
    chat.tag_config("bot", foreground="#222222")
    chat.tag_config("meta", foreground="#888888", font=("Segoe UI", 8, "italic"))

    def add(sender, text, tag, meta=None):
        chat.config(state="normal")
        chat.insert(tk.END, f"{sender}: ", tag)
        chat.insert(tk.END, text + "\n", "bot" if tag == "bot" else "user")
        if meta:
            chat.insert(tk.END, meta + "\n", "meta")
        chat.insert(tk.END, "\n")
        chat.config(state="disabled")
        chat.see(tk.END)

    def send(event=None):
        text = entry.get().strip()
        if not text:
            return
        entry.delete(0, tk.END)
        add("You", text, "user")
        answer, matched, score = bot.reply(text)
        meta = f'matched: "{matched}" ({score:.0%})' if matched else None
        add("Bot", answer, "bot", meta)
        if answer.startswith("Goodbye"):
            root.after(800, root.destroy)

    bottom = tk.Frame(root)
    bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
    entry = tk.Entry(bottom, font=("Segoe UI", 11))
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
    entry.bind("<Return>", send)
    tk.Button(bottom, text="Send", command=send, width=8).pack(side=tk.LEFT, padx=(6, 0))

    add("Bot", "Hi! Ask me anything about EduLearn, e.g. 'How do I get a refund?'", "bot")
    entry.focus()
    root.mainloop()


# ---------------------------------------------------------------------------
def main():
    print("1) Console chat\n2) Chat window (Tkinter)")
    choice = input("Choose 1 or 2: ").strip()
    gui_chat() if choice == "2" else console_chat()


if __name__ == "__main__":
    main()

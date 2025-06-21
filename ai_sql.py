import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import sys  

# Generating SQL queries
def generate_sql_query(is_injection=False):
    if is_injection:
        injections = [
            "SELECT * FROM users WHERE email = '' OR '1'='1';",
            "SELECT * FROM users WHERE email = ''; DROP TABLE users;",
            "SELECT * FROM users WHERE username = 'admin' --",
            "SELECT * FROM products WHERE id = 1; DROP TABLE orders;",
            "SELECT * FROM data WHERE value = 'x'; --"
        ]
        return random.choice(injections)
    else:
        safe_queries = [
            "SELECT * FROM users WHERE email = 'admin@example.com';",
            "SELECT name FROM employees WHERE department = 'Sales';",
            "SELECT id FROM orders WHERE date = '2023-01-01';",
            "SELECT * FROM products WHERE category = 'Electronics';",
            "SELECT COUNT(*) FROM users;",
            "SELECT * FROM customers WHERE name = 'John';",
            "SELECT * FROM messages WHERE status = 'unread';"
        ]
        return random.choice(safe_queries)

# Data preparation
queries = []
labels = []

for _ in range(50):
    for _ in range(10):
        queries.append(generate_sql_query(is_injection=True))
        labels.append(1)
    for _ in range(15):
        queries.append(generate_sql_query(is_injection=False))
        labels.append(0)

data = pd.DataFrame({'query': queries, 'label': labels})
X_train, X_test, y_train, y_test = train_test_split(data['query'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Scaner
def detect_sql():
    print("\n🧠 The SQL analyzer is active. Enter the SQL queries one at a time.")
    print("Enter 'exit' to exit manually.")

    while True:
        query = input("\nSQL > ")
        if query.lower() == "exit":
            print("👋 Manual completion.")
            break

        query_vec = vectorizer.transform([query])
        prediction = model.predict(query_vec)[0]

        # Primitive rules — helps to identify manual attacks
        keywords = ["' or '", "drop table", "--", "sleep(", "union", "select * from information_schema"]

        if prediction == 1 or any(k in query.lower() for k in keywords):
            print("⛔ MALICIOUS REQUEST DETECTED!")
            print("🔁 Return to the main menu...")
            break
        else:
            print("✅ The request is safe. Keep going...")

# Launch
if __name__ == "__main__":
    print("🔍 Checking the accuracy of the model:\n")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    detect_sql()

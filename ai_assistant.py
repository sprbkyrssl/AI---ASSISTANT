import subprocess


def check_sql():
    subprocess.run(["python", "ai_sql.py"])

def check_text():
    subprocess.run(["python", "ai_antifishing.py"])

def check_code():
    subprocess.run(["python", "ai_malware_detect.py"])

def decode_and_predict():
    subprocess.run(["python", "ai_the_decryptor.py"])


def main():
    while True:
        print("\nSelect an option:")
        print("1. Checking an SQL query")
        print("2. Checking text for phishing")
        print("3. Checking Python code for malware")
        print("4. Decrypting a string")
        print("0. Exit")

        choice = input("Enter the number: ")

        if choice == "1":
            check_sql()
        elif choice == "2":
            check_text()
        elif choice == "3":
            check_code()
        elif choice == "4":
            decode_and_predict()
        elif choice == "0":
            print("Exiting the program.")
            break
        else:
            print("Incorrect choice. Try again.")


if __name__ == "__main__":
    main()

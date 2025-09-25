from modules import programming, cybersecurity, general

def main():
    print("Welcome to Cyber AI!")
    while True:
        query = input("Ask me something (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        if 'hack' in query.lower() or 'security' in query.lower():
            print(cybersecurity.answer(query))
        elif 'code' in query.lower() or 'program' in query.lower():
            print(programming.answer(query))
        else:
            print(general.answer(query))

if __name__ == "__main__":
    main()

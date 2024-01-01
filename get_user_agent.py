import random


def get_random_user_agent():
    os_version = f"Windows NT 10.0; Win64; x64"
    webkit_version = f"AppleWebKit/{random.randint(500, 600)}.{random.randint(10, 99)}"
    chrome_version = f"Chrome/{random.randint(100, 150)}.{random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    safari_version = f"Safari/{random.randint(500, 600)}.{random.randint(10, 99)}"
    user_agent = f"Mozilla/5.0 ({os_version}) {webkit_version} (KHTML, like Gecko) {chrome_version} {safari_version}"
    return user_agent
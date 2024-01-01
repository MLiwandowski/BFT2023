import subpr
import time

# Путь к основному скрипту
main_script_path = '//home/user/BinanceTBot/main_b.py'

# Запустить основной скрипт
def start_main_script():
    subprocess.Popen(['python', main_script_path])

# Ожидать, пока основной скрипт завершится, и перезапустить его
def monitor_main_script():
    while True:
        print('Запуск основного скрипта...')
        start_main_script()
        print('Ожидание завершения основного скрипта...')
        process = subprocess.Popen(['pgrep', '-f', main_script_path], stdout=subprocess.PIPE)
        process_id = process.communicate()[0].decode('utf-8').strip()
        if process_id:
            process_id = int(process_id)
            process = subprocess.Popen(['ps', '-p', str(process_id)], stdout=subprocess.PIPE)
            process_info = process.communicate()[0].decode('utf-8')
            if 'python' in process_info and main_script_path in process_info:
                process.wait()
                print('Основной скрипт завершил выполнение.')
            else:
                print('Основной скрипт не найден. Перезапуск...')
        else:
            print('Основной скрипт не найден. Перезапуск...')
        time.sleep(60)  # Периодическая проверка каждую минуту

if __name__ == '__main__':
    monitor_main_script()

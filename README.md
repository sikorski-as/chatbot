# Deep-learning general-conversation chatbot
## Opis
Projekt mający na celu stworzenie chatbota do konwersacji o tematyce ogólnej. 
Chatbot oparty o głęboką sieć, model seq2seq.

## Autorzy
- Marcin Żyżyński
- Kacper Wnuk
- Arkadiusz Sikorski

## Wymagania
- Interpreter języka Python w wersji `>= 3.6`.
- Menadżer pakietów `pip`.

## Zależności
- Pakiet `tensorflow==2.1.0`.
- Pakiet `nltk`.

## Uruchomienie i instalacja 
W folderze z chatbotem należy uruchomić skrypt instalacyjny. W zależności od systemu operacyjnego jest to:
- `install.sh` (Linux)
- `install.bat` (Windows)

Instalator powinien utworzyć wirtualne środowiska dla pythona w folderze `bin/.venv`.
Jeśli instalacja przebiegła pomyślnie, należy uruchomić skrypt `run.sh`/`run.bat`, by rozpocząć działanie programu.

## Wybór wariantu czatbota i konwersacja
Potencjalnie można rozmawiać z różnymi czatbotami (wytrenowanych na przykład na różnych danych tekstowych).
Po starcie czatbota można wybrać jego wariant podając **nazwę** z wyświetlonej na ekranie listy.
Lista dostępnych wariantów jest tworzona na podstawie plików z katalogu `setups/`.

Po wybraniu wariantu powinna rozpocząć się konwersacja zachęcająca użytkownika do wpisania wiadomości do czatbota. 
Po kliknięciu klawisza enter, bot powinien odpowiedzieć na podstawie modelu zawartego w wybranym wariancie.

Aby zakończyć konwersację z czatbotem, wystarczy wcisnąć kombinację klawiszy `Ctrl+C`.
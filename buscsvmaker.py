from pdfminer.high_level import extract_text
import csv

pdf_file = 'swk-bus-2023-taschenfahrplan.pdf'
text = extract_text(pdf_file)

lines = []
for line in text.split('\n'):
    if line.startswith('Bahnhof,•') or line.startswith('Winter') or line.startswith('Sommer'):
        lines.append(line)

with open('swk-bus-2023-taschenfahrplan.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['station', 'linenr', 'weekday', 'time'])
    for line in lines:
        if line.startswith('Bahnhof,•'):
            station = line.split(',')[1].strip()
            linenr = line.split(',')[2].strip()
            weekday = 'Mon-Fri'
            time = line.split(',')[3:]
            writer.writerow([station, linenr, weekday, ' '.join(time)])
        elif line.startswith('Winter') or line.startswith('Sommer'):
            weekday = 'Sat-Sun'
            linenr = ''
            station = ''
            time = line.split()
            writer.writerow([station, linenr, weekday, ' '.join(time)])

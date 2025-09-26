Q: agnesi пшен
gold: [(0, 6, 'B-BRAND'), (7, 11, 'B-TYPE')]
raw: [('B-BRAND', 0, 2), ('I-BRAND', 2, 6), ('B-TYPE', 7, 8), ('I-TYPE', 8, 11)]
pp : [(0, 2, 'B-BRAND'), (2, 6, 'I-BRAND'), (7, 8, 'B-TYPE'), (8, 11, 'I-TYPE')]
NOT OK

Q: ahma
gold: [(0, 4, 'B-BRAND')]
raw: []
pp : []
NOT OK

Отлично. Теперь давай пройдемся по строкам где source = s и длина текста 4 символа и установлен B-BRAND, а так же текст элемента ниже первые 3 символа sample не равны sample текущей строки. Для таких строк поставим rules4
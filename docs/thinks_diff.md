sample: вермишель быстро приготовления
  gold: [(0, 9, 'B-TYPE'), (10, 16, 'O'), (17, 30, 'I-TYPE')]
  pred: [(0, 9, 'B-TYPE'), (10, 16, 'I-TYPE'), (17, 30, 'I-TYPE')]

быстрого - болжно быть на I-TYPE

sample: рте
  gold: [(0, 3, 'B-TYPE')]
  pred: [(0, 3, 'O')]

скорее всего O

sample: сыи
  gold: [(0, 3, 'B-TYPE')]
  pred: [(0, 3, 'O')]

скорее всего O

sample: нф калькуляторы
  gold: [(0, 2, 'B-BRAND'), (3, 15, 'B-TYPE')]
  pred: [(0, 2, 'O'), (3, 15, 'B-TYPE')]

нф - O

sample: сок сады вишня
  gold: [(0, 3, 'B-TYPE'), (4, 8, 'B-BRAND'), (9, 14, 'I-BRAND')]
  pred: [(0, 3, 'B-TYPE'), (4, 8, 'B-BRAND'), (9, 14, 'O')]

вишня - I-TYPE


sample: джениус подскажи
  gold: [(0, 7, 'B-BRAND'), (8, 16, 'B-TYPE')]
  pred: [(0, 7, 'O'), (8, 16, 'B-TYPE')]

В train такого нет - думаю нужно занулять

sample: lph
  gold: [(0, 3, 'B-BRAND')]
  pred: [(0, 3, 'O')]

Занулять

все для праздника;[(0, 3, 'B-TYPE'), (4, 7, 'O'), (8, 17, 'O')] - BRAND ? В train занулено

chco;[(0, 4, 'B-BRAND')] - занулять

rri;[(0, 3, 'B-BRAND')] - занулять

доя мытья посуды;[(0, 3, 'B-TYPE'), (4, 9, 'I-TYPE'), (10, 16, 'I-TYPE')] - занулять

sample: хмели сун
  gold: [(0, 5, 'B-TYPE'), (6, 9, 'B-BRAND')]
  pred: [(0, 5, 'B-TYPE'), (6, 9, 'I-TYPE')]

  все бренд

sample: вода минер
  gold: [(0, 4, 'B-TYPE'), (5, 10, 'B-BRAND')]
  pred: [(0, 4, 'B-TYPE'), (5, 10, 'I-TYPE')]

Все Type

вый;[(0, 3, 'B-TYPE')]

Занулять


sample: корм кош сухой
  gold: [(0, 4, 'B-TYPE'), (5, 8, 'O'), (9, 14, 'I-TYPE')]
  pred: [(0, 4, 'B-TYPE'), (5, 8, 'I-TYPE'), (9, 14, 'I-TYPE')]

Все TYPE


sample: капуста свежий урожай
  gold: [(0, 7, 'B-TYPE'), (8, 14, 'B-BRAND'), (15, 21, 'O')]
  pred: [(0, 7, 'B-TYPE'), (8, 14, 'I-TYPE'), (15, 21, 'B-BRAND')]
  mismatch_spans: [(8, 14), (15, 21)]

TYPE + BRAND



томаты в томатном соусе;[(0, 6, 'B-TYPE'), (7, 8, 'O'), (9, 17, 'O'), (18, 23, 'I-TYPE')]

Type + O


в томатном соусе рыьа;[(0, 1, 'O'), (2, 10, 'O'), (11, 16, 'O'), (17, 21, 'I-TYPE')]

O + Type. Перепроверить, возможно все занулить


фасоль в томатном соусе;[(0, 6, 'B-TYPE'), (7, 8, 'O'), (9, 17, 'O'), (18, 23, 'O')]

TYPE + O


sample: угле поле
  gold: [(0, 4, 'B-TYPE'), (5, 9, 'I-TYPE')]
  pred: [(0, 4, 'B-TYPE'), (5, 9, 'B-BRAND')]

все BRAND


sample: ягодное лукошко
  gold: [(0, 7, 'B-TYPE'), (8, 15, 'I-TYPE')]
  pred: [(0, 7, 'B-TYPE'), (8, 15, 'B-BRAND')]

ВСЕ BRAND


вд питьея;[(0, 2, 'B-BRAND'), (3, 9, 'B-TYPE')]

TYPE
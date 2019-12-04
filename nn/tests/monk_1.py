_monk_name = 'monks-1'

with open('../../monks/' + _monk_name + '.train') as f:
    train_data = f.readlines()
train_data = [line.split(' ') for line in train_data]
train_data = tuple(map(
    lambda el: (
        tuple(map(lambda lx: float(lx), el[2:-1])),
        [float(el[1])]),
    train_data))

with open('../../monks/' + _monk_name + '.test') as f:
    test_data = f.readlines()
test_data = [line.split(' ') for line in test_data]
test_data = tuple(map(
    lambda el: (
        tuple(map(lambda x: float(x), el[2:-1])),
        [float(el[1])]),
    test_data))

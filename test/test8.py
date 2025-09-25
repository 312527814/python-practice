def my_function(**kwargs):
    print(kwargs)
    for key, value in kwargs.items():
        print(f"{key}: {value}")

data = {'name': 'Alice', 'age': 30, 'city': 'New York'}
# my_function({'name':'dd'})
aa={'fname':"tuobiasi"}
my_function(**aa)

print(type(data))
# my_function(data)
import numpy as np
import random

def generateRandomList(list_size):
	return np.random.rand(list_size, 2)

batch_size = 4
max_list_Size = 16

lista = np.zeros((max_list_Size, 2)) 

print(lista)
ptr = 0



for i in range(20):

	lista[ptr:ptr+batch_size] = generateRandomList(batch_size)
	#np.put(lista, range(ptr, ptr+batch_size), generateRandomList(batch_size))
	print(lista)


	ptr = (ptr + batch_size) % max_list_Size
	print("new ptr value ", ptr)
	print("mod ", ptr % max_list_Size )



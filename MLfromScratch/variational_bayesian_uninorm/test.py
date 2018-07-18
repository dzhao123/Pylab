import numpy as np


rho_nk = np.random.rand(3, 3)
r_nk = rho_nk / np.sum(rho_nk,1)

#print(rho_nk)
#print(r_nk)

#print(np.matmul([[1,2],[3,4]],[[1,2],[3,4]]))
#print(np.sum([[1,2],[3,4]],0,keepdims=True))
#print(np.array([[1,2],[3,4]])/np.array([[4],[6]]))
#print(np.array([1,2,3,4])[:,None])

#print(np.zeros(3))
#print(np.array([2,3])*np.array([[[1,2],[3,4]],[[1,2],[3,4]]]))
#print(3*np.array([[1,2],[3,4]]))
#print(np.dot(np.matmul([[1,2],[3,4]],[5,6]),[5,6]))
#print(np.ones(5))
print(np.expand_dims([1,2,3],1))

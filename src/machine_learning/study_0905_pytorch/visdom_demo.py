import numpy as np
import scipy.sparse as sp

def CSR(x):#输入一个矩阵
    w=len(x[0])
    h=len(x)
    A = np.array(x)
    AS = sp.csr_matrix(A)
    print("data=",AS.data)
    print("indptr=",AS.indptr)
    print("indices=",AS.indices)
    print("compress_rate=",(len(AS.data)+len(AS.indptr)+len(AS.indices))/(w*h))
    #输出矩阵的CSR压缩率

#随机生成一个方阵，输入矩阵的尺寸，秩的上限值，稀疏度
def generate_matrix(x,r,s): #size=x*x,rank<=r,sparsity=s
    matrix=[[0]*x]*x
    matrix_r=np.random.randint(0,100,(r,x))
    matrix_r=matrix_r.tolist()
    for i in range(len(matrix_r)):
        for j in range(len(matrix_r[0])):
            if(matrix_r[i][j]<100*s):
                matrix_r[i][j]=0
    for k in range(x):
        if(k<r):
            matrix[k]=matrix_r[k]
        else:
            matrix[k]=matrix[k-r]
    return matrix

a=generate_matrix(6,6,0.8)#稀疏度为0.8的矩阵
for i in a:
    print(i)
CSR(a)
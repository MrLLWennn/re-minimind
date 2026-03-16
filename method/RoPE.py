import torch

# tensor
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])

#where
# condition = (x > 3)

# result1 = torch.where(condition, x, y)
#print(result1)

#arrange
#print(torch.arange(10,60,10))

#outer
#print(torch.outer(x,y))
#print(torch.outer(y,x))

#shape cat
# t1 = torch.tensor([[1, 2], 
#                    [3, 4]])
# t2 = torch.tensor([[5, 6], 
#                    [7, 8]])
# print(t1.shape)
# print(t2.shape)
# t3 = torch.cat((t1, t2), dim=1)  # Concatenate along rows
# print(t3.shape)
# print(t3)
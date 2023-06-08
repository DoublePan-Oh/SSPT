import torch
# a2 = torch.rand(2,3) # shape=(2, 3) 下同
# a3 = torch.rand(2,3)
# t2_stack_0 = torch.stack([a2, a3], axis=0)  # shape=(2,2,3)
# t2_stack_1 = torch.stack([a2, a3], axis=1) # shape=(2,2,3)

# print(t2_stack_0.shape, t2_stack_1.shape)
# print(t2_stack_0, t2_stack_1)
y = torch.rand(2, 2, 4)
x = (torch.rand(2, 2, 1) > 0.1).float()
print(x)
print(y)
print((y * x))
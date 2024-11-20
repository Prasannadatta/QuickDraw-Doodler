
import torch


if __name__ == '__main__':
    a, b = torch.rand((3,4)), torch.rand((3,4))
    print(a+b, '\n', a-b)
    print(torch.dot(a.view(-1), b.view(-1)))
    print(torch.matmul(a, b.T))

    c, d = torch.rand((256,28*28)), torch.rand((28*28, 10))
    print(torch.matmul(c, d).shape)

    print('*****************')

    a = torch.rand(3, requires_grad=True)
    b = torch.rand(3, requires_grad=True)
    c = torch.rand(3, requires_grad=True)
    print(f"Inputs:\n{a}\n{b}\n{c}\n")
    d = (a+b)
    e = d*c
    print(f"Grad fns:\n{d}\n{e}\n")

    torch.sum(e).backward()
    print(f"Gradients:\n{a.grad}\n{b.grad}\n{c.grad}")

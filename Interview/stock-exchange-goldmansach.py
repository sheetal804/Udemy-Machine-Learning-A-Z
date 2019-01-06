from bisect import bisect

n = int(input())
asks = sorted(zip(
    map(int, input().split()),
    map(int, input().split())
))

for _ in range(int(input())):
    res=[]
    q = int(input())
    idx = bisect(asks, (q, float('inf')))
    print(asks)
    print(asks[idx-1][1])
    res.append(asks[idx-1][1])

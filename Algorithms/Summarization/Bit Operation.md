# Bit Operation

Set/Clear value to bit-th bit

```python []
def set_bit(value, bit) -> int:
    return value | (1<<bit)

def clear_bit(value, bit) -> int:
    return value & ~(1<<bit)
```

Determine if bit-bit is 1

```python
def if_is_one(value, bit) -> bool:
    return value >> bit & 1
```


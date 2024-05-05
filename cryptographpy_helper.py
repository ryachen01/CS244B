from sympy import isprime, randprime
import hashlib
import hmac

def find_prime_of_order_q(q):
    """Find a prime p = kq + 1."""
    k = 1
    while True:
        p = k * q + 1
        if isprime(p):
            return p
        k += 1

def find_generator(p, q):
    """Find a generator of the cyclic group of order q modulo p."""
    for g in range(2, p - 1):
        if pow(g, q, p) == 1 and pow(g, (p - 1) // q, p) != 1:
            return g
    return None

def generate_cyclic_group(lambda_bits):
    """Generate a cyclic group G of order q with bit length ||q||"""
    q = randprime(2**(lambda_bits - 1), 2**lambda_bits)
    p = find_prime_of_order_q(q)
    g = find_generator(p, q)
    if g is None:
        return None  
    return (p, q, g)

def hmac_digest(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if len(salt) == 0:
        salt = bytes([0] * hashlib.sha256().digest_size)
    return hmac_digest(salt, ikm)


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    t = b""
    okm = b""
    i = 0
    while len(okm) < length:
        i += 1
        t = hmac_digest(prk, t + info + bytes([i]))
        okm += t
    return okm[:length]


def hkdf(salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)



from sympy import isprime, randprime
import random
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


def eval_polynomial(coeffs, x, p):
    """Evaluate a polynomial with given coefficients at x modulo p."""
    return sum((c * pow(x, i, p)) % p for i, c in enumerate(coeffs)) % p

def generate_shares(secret, n, k, p):
    """Generate n shares with threshold k from the secret."""
    # Randomly generate coefficients for a polynomial of degree k-1
    coeffs = [secret] + [random.randint(0, p-1) for _ in range(k-1)]
    shares = [(i, eval_polynomial(coeffs, i, p)) for i in range(1, n+1)]
    return shares

def lagrange_interpolation(x, points, p):
    """Perform Lagrange interpolation to find the polynomial at x."""
    def basis(j):
        num = den = 1
        for m in range(len(points)):
            if m != j:
                num = (num * (x - points[m][0])) % p
                den = (den * (points[j][0] - points[m][0])) % p
        return (num * pow(den, -1, p)) % p
    
    return sum((points[j][1] * basis(j)) % p for j in range(len(points))) % p

def recover_secret(shares, p):
    """Recover the secret from shares using Lagrange interpolation."""
    return lagrange_interpolation(0, shares, p)

# lambda_bits = 8
# p, q, g = generate_cyclic_group(lambda_bits)

# import secrets
# import numpy as np

# num_clients = 5
# secret_keys = [secrets.randbelow(q) for _ in range(num_clients)]
# public_keys = [pow(g, secret_keys[i], p) for i in range(num_clients)]

# common_keys = [[None]*num_clients for _ in range(num_clients)]
# for i in range(num_clients):
#     for j in range(num_clients):
#         common_keys[i][j] = pow(public_keys[j], secret_keys[i], p)

# print(np.array(common_keys))

# secret = random.randint(0, p-1)
# n = 5  # Total number of shares
# k = 3  # Threshold number of shares needed to reconstruct the secret

# shares = generate_shares(secret, n, k, p)
# print("Shares:", shares)

# # Simulate recovering the secret using any k shares
# recovered_secret = recover_secret(shares[:k], p)
# print("Recovered secret:", recovered_secret)
# print("Secret is correct:", recovered_secret == secret)
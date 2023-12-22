export function invert(permutation: number[]) {
    const result = new Array(permutation.length);
    for (let i = 0; i < permutation.length; i++) {
        result[permutation[i]] = i;
    }
    return result;
}
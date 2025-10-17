# Analysis of Remaining 30 Errors

## Current State
- Total errors: 28-30 (exact count varies by build)
- Progress: 83.5% complete (182→30)
- All infrastructure: FIXED ✅
- Remaining: Borrow checker + edge cases

## Error Categories

### 1. E0502 - Borrow Conflicts (6 errors - 21%)
**Pattern**: Cannot borrow as mutable because also borrowed as immutable

**Likely locations**:
- Methods with multiple self borrows
- Iterator + mutation patterns
- Nested lock/borrow situations

**Fix approach**: Restructure to avoid overlapping borrows

### 2. E0382 - Moved Values (6 errors - 21%)
**Pattern**: Use of moved value

**Known issues from report**:
- ccm_result
- causal_matrix  
- sqrt_matrix
- witness
- error
- x

**Fix approach**: Add .clone() or restructure ownership

### 3. E0308 - Type Mismatches (5 errors - 18%)
**Pattern**: Mismatched types

**Fix approach**: Type conversions, casts, proper constructors

### 4. E0599 - Missing Methods (3 errors - 11%)
**Pattern**: `.launch()` not found on CudaFunction

**Fix approach**: Use proper cudarc launch API

### 5. E0061 - Argument Count (3 errors - 11%)
**Pattern**: Function takes N arguments but M supplied

**Known issue**: BPETokenizer calls have wrong arg counts

**Fix approach**: Check API signatures and adjust

### 6. E0507 - Move from Shared Ref (2 errors - 7%)
**Pattern**: Cannot move out of shared reference

**Known issue**: `agent.beliefs.mu`

**Fix approach**: Clone instead of move

### 7. E0596 - Closure Mut (1 error - 4%)
**Pattern**: Closure needs to be mutable

**Fix approach**: `let mut closure = ||`

### 8. E0432 - Import (1 error - 4%)
**Pattern**: `crate::pwsa` unresolved

**Fix approach**: Feature gate the import

### 9. E0277 - Iterator (1 error - 4%)
**Pattern**: Vec<Complex64> cannot be built from iterator

**Fix approach**: Explicit collect with type annotation

## Specific Error Details


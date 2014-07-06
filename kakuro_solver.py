#!/usr/bin/env python
import math
import sys
import logging
import hashlib

solution = {}
solutions = {}
partial = {}
unique_runs_h = []
unique_runs_v = []
horizontal_runs = {}
vertical_runs = {}

class Memoize:
  def __init__(self, f):
    self.f = f
    self.memo = {}

  def __call__(self, *args):
    if not args in self.memo:
      self.memo[args] = self.f(*args)
    return self.memo[args]


def combinations(nums):
  combos = []
  if len(nums) == 2:
    return [[nums[0], nums[1]], [nums[1], nums[0]]]
  else:
    for num in nums:
      nums2 = nums[:]
      nums2.remove(num)
      combos.extend([x + [num] for x in combinations(nums2)])
  return combos


def parse_sequence(idx, cells, run_dict, vert):
  total = 0
  start = []
  length = 0
  for pitch, cell in enumerate(cells):
    if cell.isdigit():
      if int(cell) > 0:
        if total != 0:
          add_run(start, length, total, run_dict, vert)
        total = int(cell)
        length = 0
        start = [idx, pitch + 1] if vert else [pitch + 1, idx]
      elif int(cell) == 0:
        length += 1
  if total != 0:
    add_run(start, length, total, run_dict, vert)

def add_solution(remaining=0):
  m = hashlib.md5()
  m.update(str(solution.values()))
  if m.hexdigest() not in solutions:
    if remaining > 0:
      partial[m.hexdigest()] = remaining
    solutions[m.hexdigest()] = solution.copy()
    return True
  return False

def add_run(start, length, total, run_dict, vert):
  if length == 0:
    raise Exception('Error in puzzle at: %s' % (start,))
  run = Run(total, length, start, vert)
  idx = 1 if vert else 0
  if vert:
    unique_runs_v.append(run)
  else:
    unique_runs_h.append(run)
  for _ in range(length):
    run_dict[tuple(start)] = run
    start[idx] += 1


def inv_triangular(coeff):
  return int(-0.5 + math.sqrt(0.25 + 2*coeff))


@Memoize
def find_sequences(total, length):
  sequences = []
  if total == 3 and length == 2:
    return [[1, 2]]
  if length == 1:
    return [[total]]
  start = min((total - 1, 9))
  while start > 0:
    diff = total - start
    spaces = inv_triangular(diff) + 1
    start -= 1
    if spaces < length:
      continue
    sub_sequences = find_sequences(diff, length - 1)
    for sub_sequence in sub_sequences:
      if len(sub_sequence) < length and (start + 1) not in sub_sequence:
        sequence = sub_sequence[:]
        sequence.append(start + 1)
        sequence = sorted(sequence)
        if sequence not in sequences and sequence[-1] <= 9:
          sequences.append(sequence)
  return sequences


def solve(width, height):
  global solution
  iterations = 0
  current_filled = 1
  limit = 2
  while len(solution) < len(horizontal_runs) and iterations < 45:
    all_tested = True
    iterations += 1
    last_filled = current_filled
    current_filled = 0
    for run in unique_runs_h + unique_runs_v:
      current_filled += run.fill_cells()
    if last_filled == 0:
      for run in unique_runs_h + unique_runs_v:
        all_tested &= run.test_possibilities(limit)
      if all_tested:
        if Run.min_remaining > 0:
          for partial_solution in solutions:
            if len(partial_solution) == Run.min_remaining:
              solution = partial_solution
        else:
         break
      limit *= 2
  logging.debug("Limit: %s" % limit)
  logging.debug("Iterations: %s" % iterations)
  if len(solution) != len(horizontal_runs):
    print "No unique solution found."
  else:
    Run.min_remaining = 0
    add_solution()
  print_solutions(width, height)

def print_solutions(width, height):
  idx = 1
  for key in solutions:
    if key in partial and Run.min_remaining != partial[key]:
      continue
    print "Solution %s:" % idx
    for y in range(height):
      for x in range(width):
        if (x, y) in solutions[key]:
          print "%i " % solutions[key][(x, y)],
        elif (x, y) not in solutions[key] and (x, y) in horizontal_runs:
          print "X ",
        else:
          print "# ",
      print
    idx += 1


class Run(object):
  coord_changes = []
  h_guess_coords = {}
  v_guess_coords = {}
  min_remaining = 0

  def __init__(self, total, length, start, vert):
    self.length = length
    self.total = total
    self.intersect = horizontal_runs if vert else vertical_runs
    self.__class__.min_remaining = len(horizontal_runs)
    self.sequences = find_sequences(total, length)
    if len(self.sequences) == 0:
      raise Exception("Error at %s, no digit sequences found" % (start,))
    self.digit_coords = {x: set() for x in range(1, 10)}
    (a, b) = (0, 1) if vert else (1, 0)
    self.coords = [(start[0] + a * x, start[1] + b * x) for x in range(length)]

  def get_digits(self):
    found_digits = set(self._get_found())
    all_digits = set()
    required_digits = set(self.sequences[0])
    for sequence in self.sequences:
      if found_digits.issubset(set(sequence)) or len(found_digits) == 0:
        all_digits = all_digits | set(sequence)
        required_digits = required_digits & set(sequence)
    all_digits = all_digits - found_digits
    required_digits = required_digits - found_digits 
    return all_digits, required_digits

  def _get_found(self):
    found_digits = []
    for coord in self.coords:
      if coord in solution:
        found_digits.append(solution[coord])
    return found_digits

  def undo(self, guess_coord):
    coord = None
    while coord != guess_coord:
      coord = self.coord_changes.pop()
      value = solution[coord]
      del solution[coord]

  def add_found(self, coord, found, testing=False):
    solution[coord] = found
    if not testing:
      if coord in self.h_guess_coords:
        del self.h_guess_coords[coord]
      if coord in self.v_guess_coords:
        del self.v_guess_coords[coord]
    self.coord_changes.append(coord)

  def test_possibilities(self, limit):
    all_, required = self.get_digits()
    combos = []
    if len(all_) == len(required) and len(all_) > 0:
      combos = combinations(list(required))
    else:
      remaining = self.total
      length = self.length
      for coord in self.coords:
        if coord in solution:
          remaining -= solution[coord]
          length -= 1
      if length == 2:
        sub_sequences = find_sequences(remaining, length)
        for sequence in sub_sequences:
          combos.extend(combinations(sequence))
    if len(combos) <= limit and len(combos) != 0:
      self._test(combos)
    return (len(combos) <= limit)

  def _fill_unique(self, required_digits):
    count = 0
    for digit, coords in self.digit_coords.iteritems():
      if len(coords) == 1 and digit in required_digits:
        coord = coords.pop()
        if coord not in solution:
          count += 1
          logging.debug("Adding: %s %s" % (coord, digit))
          self.add_found(coord, digit)
    return count

  def fill_cells(self, test=False):
    self.digit_coords = {x: set() for x in range(1, 10)}
    found = self._get_found()
    if len(found) != len(set(found)):
      return -1
    digits1, digits2 = self.get_digits()
    filled_count = 0
    for coord in self.coords:
      if coord not in solution:
        digits3, digits4 = self.intersect[coord].get_digits()
        common = digits3 & digits1
        if len(common) == 1:
          found = common.pop()
          logging.debug("Found: %s %s" % (coord, found))
          self.add_found(coord, found, test)
          if found in digits2:
            digits2.remove(found)
          filled_count += 1
        elif len(common) == 0:
          return -1
        for digit in common:
          self.digit_coords[digit].add(coord)
        if test and filled_count != 0 and self.intersect[coord].fill_cells(test) == -1:
          return -1
    filled_count += self._fill_unique(digits2)
    return filled_count

  def _test(self, value_set):
    valid = []
    for values in value_set:
      idx = 0
      eliminated = False
      test_coords = []
      for run_coord in self.coords:
        if run_coord not in solution:
          test_coords.append(run_coord)
          self.add_found(run_coord, values[idx])
          idx += 1
      for test_coord in test_coords:
        if self.intersect[test_coord].fill_cells(True) == -1:
          eliminated = True
          break
      if not eliminated:
        valid.append(values)
      if len(self.intersect) == len(solution):
        add_solution()
        self.__class__.min_remaining = 0
      elif (len(self.intersect) - len(solution)) <= self.min_remaining:
        self.__class__.min_remaining = (len(self.intersect) - len(solution))
        add_solution(self.min_remaining)
      self.undo(test_coords[0])
    if len(valid) == 1:
      logging.debug("Adding: %s %s" % (self.coords[0], valid[0]))
      idx = 0
      for run_coord in self.coords:
        if run_coord not in solution:
          self.add_found(run_coord, valid[0][idx])
          idx += 1


if __name__ == "__main__":
  vert = False
  columns = []
  width = 0
  height = 0
  if len(sys.argv) == 3 and sys.argv[2] == '--debug':
    logging.basicConfig(level=logging.DEBUG, format="Debug: %(message)s", stream=sys.stdout)
  with open(sys.argv[1], 'r') as in_file:
    for line_no, line in enumerate(in_file):
      if len(line.strip()) == 0:
        vert = True
      else:
        cells = line.strip().split(',')
        width = len(cells)
        if vert:
          if len(columns) == 0:
            columns = [[] for _ in cells]
          for idx, cell in enumerate(cells):
            columns[idx].append(cell)
        else:
          parse_sequence(line_no, cells, horizontal_runs, vert)
  height = len(columns[0])
  for idx, column in enumerate(columns):
    parse_sequence(idx, column, vertical_runs, vert)
  try:
    solve(width, height)
  except:
    solutions[0] = solution
    print_solutions(width, height)
    raise

#pragma once

template<class T>
void printVector(const std::vector<T>& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << v[i] << ", ";
  std::cout << "\n";
}

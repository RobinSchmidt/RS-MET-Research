
/** This version gets instantiated when there is only a single template parameters. it serves as 
recursion anchor. */
template <typename T>
void printLines(T line)
{
  std::cout << line << "\n";
}

//template <typename T>
//void printLines()
//{
//  // recursion anchor which gets instantiated when the template argument list is empty
//}

template<typename First,typename ... Rest>
void printLines(First first,Rest ... rest)
{
  printLines(first);
  printLines(rest...);         // recursive instantiation
}
// code belongs to this video: https://www.youtube.com/watch?v=9rZMqwW52D0

#include <iostream>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <vector>
#include <deque>

// Euclidean Algorithm
// returns greatest common divisor
template<typename T>
T gcd(T a, T b)
{
	if (b > a) std::swap(a, b);

	T r;
	while ((r = a%b) != 0)
	{
		a = b; b = r;
	}

	return b;
}

// least common multiple
template<typename T>
T lcm(T a, T b)
{
	T G = gcd(a, b);

	return (a / G)*b;
}

template<typename T>
class UnsignedRational
{
public:
	using value_type = std::make_unsigned_t<T>;
private:
	value_type m_n{ 0 }; // numerator
	value_type m_d{ 1 }; // denominator

	void reduce()
	{
		value_type G = gcd(this->m_n, this->m_d);

		this->m_n /= G; this->m_d /= G;
	}

public:

	UnsignedRational() { }
	UnsignedRational(T n, T d) : m_n(n), m_d(d) { reduce(); }

	UnsignedRational(T n, T d, bool) :
		m_n(n), m_d(d) { }

	UnsignedRational(const UnsignedRational& rhs) = default;
	UnsignedRational& operator=(const UnsignedRational& rhs) = default;

	UnsignedRational(UnsignedRational&& rhs) = default;
	UnsignedRational& operator=(UnsignedRational&& rhs) = default;

	~UnsignedRational() = default;

	template<typename C>
	operator C() const
	{
		return C((double)this->m_n / (double)this->m_d);
	}

	void operator *= (const UnsignedRational& r)
	{
		// m_n     r.m_n         m_p        r.m_p
		//----- * -------- = (-------) * (-------)
		// m_d     r.m_d        r.m_d        m_d

		UnsignedRational r1(this->m_n, r.m_d);
		UnsignedRational r2(r.m_n, this->m_d);

		this->m_n = r1.m_n * r2.m_n;
		this->m_d = r1.m_d * r2.m_d;


	}

	void operator += (const UnsignedRational& r)
	{
		// m_n     r.m_n     m_n * (lcm/m_d) + r.m_n * (lcm/r.m_d)
		//----- + -------- = ------------------------------
		// m_d     r.m_d                   lcm

		T n = this->m_n, d = this->m_d;

		T L = lcm(m_d, r.m_d);

		this->m_n = n * L / d + r.m_n * (L / r.m_d);
		this->m_d = L;

		reduce();
	}

	friend std::ostream& operator<<(std::ostream& os, const UnsignedRational& r)
	{
		if (r.m_d == UnsignedRational::value_type(1))
			os << r.m_n;
		else
			os << r.m_n << "/" << r.m_d;

		return os;
	}
};

// if error, returns 0
template<typename T>
T combi_rational(T n, T r)
{
	if (r > n) return 0;
	else if (r == 0 || n == r)
		return 1;
	else if (r == (n - 1) || r == 1)
		return n;
	else
	{
		if (r > (n - r)) r = n - r;

		UnsignedRational<T> rlt{ n, r };

		for (T i = 1; i < r; ++i)
			rlt *= UnsignedRational<T>(n - i, r - i);

		return (T)rlt;
	}
}

// if error, returns 0
template<typename T>
T combi_double(T n, T r)
{
	if (r > n) return 0;
	else if (r == 0 || n == r)
		return 1;
	else if (r == (n - 1) || r == 1)
		return n;
	else
	{
		if (r > (n - r)) r = n - r;

		double rlt = 1;

		for (T i = 0; i < r; ++i)
			rlt *= double(n - i) / double(r - i);

		return T(rlt);
	}
}

template<typename T>
void _enum_combi_recursion_(std::vector<T>& v, std::deque<T>& ele, T r, T mth)
{
	if (r <= 0) return;
	else if (size_t(r) == ele.size())
	{
		for (auto& e : ele)
			v.emplace_back(e);
	}
	else
	{
		T count = combi_rational(T(ele.size() - 1), r - 1);
		
		if (mth < count)
		{
			v.emplace_back(ele[0]);
			ele.pop_front();
			_enum_combi_recursion_(v, ele, r - 1, mth);
		}
		else
		{
			mth -= count;
			ele.pop_front();
			_enum_combi_recursion_(v, ele, r, mth);
		}
	}
}

template<typename T>
std::vector<T> enum_combi_recursion(T n, T r, T mth)
{
	std::deque<T> ele((size_t)n);
	for (size_t i = 0; i < size_t(n); ++i)
		ele[i] = T(i);
	std::vector<T> v;
	v.reserve((size_t)r);

	_enum_combi_recursion_(v, ele, r, mth);

	return v;
}

template<typename T>
std::vector<T> enum_combi_loop(T n, T r, T mth)
{
	std::deque<T> ele((size_t)n);
	for (size_t i = 0; i < size_t(n); ++i)
		ele[i] = T(i);

	std::vector<T> v;
	v.reserve((size_t)r);

	do
	{
		if (r <= 0) 
			break;
		else if (size_t(r) == ele.size())
		{
			for (auto& e : ele)
				v.emplace_back(e);
			
			break;
		}
		else
		{
			T count = combi_rational(T(ele.size() - 1), r - 1);

			if (mth < count)
			{
				v.emplace_back(ele[0]);
				ele.pop_front(); --r;
			}
			else
			{
				mth -= count;
				ele.pop_front();
			}
		}

	} while (true);

	return v;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	if (v.empty()) return os;

	auto last_element = --v.end();
	os << "{";
	for (auto& itr = v.begin(); itr != last_element; ++itr)
		os << *itr << ", ";

	os << *last_element << "}";

	return os;
}

int main()
{
	intmax_t n;
	intmax_t r;

	intmax_t cmb;

again:
	std::cout << "n = ?"; std::cin >> n;
	std::cout << "r = ?"; std::cin >> r;
	
	if (n == 0 && r == 0)
		return 0;

	cmb = combi_rational(n, r);
	
	for (intmax_t mth = 0; mth < cmb; ++mth)
	{
		std::cout << mth << " --- \nRecursion: ";
		std::cout << enum_combi_recursion(n, r, mth) << std::endl;
		std::cout <<"Loop     : "<<enum_combi_loop(n, r, mth) << std::endl;
	}

	goto again;
	
	return 0;

}


#include <iostream>
#include <map>
#include <set>
#include <list>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <deque>
#include <queue>
#include <stack>
#include <limits>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <variant>
#include <optional>
#include <tuple>
#include <format>
#include <sstream> 

//Type of order
enum class OrderType
{
	GoodTillCancel,
	FillAndKill
};

//Side
enum class Side
{
	Buy,
	Sell
};

//alias integers for clarity
using Price = std::int32_t;
using Quantity = std::uint32_t;
using OrderId = std::uint64_t;

//struct to call in api
struct LevelInfo
{
	Price price_;
	Quantity quantity_;
};
//aliasing the struct
using LevelInfos = std::vector<LevelInfo>;

//Object for level info for bids and asks  //internal state of Orderbook
class OrderbookLevelInfos
{
	public:
		OrderbookLevelInfos(const LevelInfos& bids, const LevelInfos& asks)
			: bids_{ bids }  //efficient member initializer list
			, asks_{ asks }
		{ }
		
		//public api's to get bids and asks //second const to make sure function does not modify any member variables of class //returned object can not be modified
		const LevelInfos& GetBids() const { return bids_; }
		const LevelInfos& GetAsks() const { return asks_; }
		
	private:
		//private api's
		LevelInfos bids_;
		LevelInfos asks_;
};

//Class for order objects including Type, Id, Side, Price, Quantity
//includes api's to get Type/Id/Side/Price/Quantity/if filled and fill command
class Order
{
	public:
		Order(OrderType orderType, OrderId orderId, Side side, Price price, Quantity quantity)
			: orderType_{ orderType }  //efficient member initializer list
			, orderId_{ orderId }
			, side_{ side }
			, price_{ price }
			, initialQuantity_{ quantity } //need to track 2 out of 3 quantities: initial, filled, remaining
			, remainingQuantity_{ quantity }
		{ }
		
		//public api's 
		OrderId GetOrderId() const { return orderId_; }
		Side GetSide() const { return side_; }
		Price GetPrice() const { return price_; }
		OrderType GetOrderType() const { return orderType_; }
		Quantity GetInitialQuantity() const { return initialQuantity_; }
		Quantity GetRemainingQuantity() const { return remainingQuantity_; }
		Quantity GetFilledQuantity() const { return GetInitialQuantity() - GetRemainingQuantity(); }
		bool IsFilled() const { return GetRemainingQuantity() == 0; }
		void Fill(Quantity quantity)
		{
			if (quantity > GetRemainingQuantity())
				throw std::logic_error(std::format("Order ({}) cannot bet filled for more than its remaining quantity", GetOrderId())); //need C++ 20
			remainingQuantity_ -= quantity;
		}
		
	private:
		//private api's
		OrderType orderType_;
		OrderId orderId_;
		Side side_;
		Price price_;
		Quantity initialQuantity_;
		Quantity remainingQuantity_;
};

//smart pointer to avoid memory leak and thread savety in reference count
using OrderPointer = std::shared_ptr<Order>; //shared because order is stored in orders dictionary and bid/ask dictionary
using OrderPointers = std::list<OrderPointer>; //list/iterator //TODO make vector, list dispersed in memory (inefficient) vector continuous in memory (efficient)

//order modify basically just cancel old order and replace with new order
class OrderModify
{
	public:
		OrderModify(OrderId orderId, Side side, Price price, Quantity quantity)
			: orderId_{ orderId }
			, price_{ price }
			, side_{ side }
			, quantity_{ quantity }
		{ }
	
	//public api's
	OrderId GetOrderId() const { return orderId_; }
	Price GetPrice() const { return price_; }
	Side GetSide() const { return side_; }
	Quantity GetQuantity() const { return quantity_; }
	
	//convert exsiting order to modified order
	OrderPointer ToOrderPointer(OrderType type) const
	{
		return std::make_shared<Order>(type, GetOrderId(), GetSide(), GetPrice(), GetQuantity());
	}
	
	//private api's
	private:
		OrderId orderId_;
		Price price_;
		Side side_;
		Quantity quantity_;
};

//Trade object (bid/ask side individually)
struct TradeInfo
{
	OrderId orderId_;
	Price price_;
	Quantity quantity_;
};

//combination of TradeInfo (bid and ask together)
class Trade
{
	public:
		Trade(const TradeInfo& bidTrade, const TradeInfo& askTrade)
			: bidTrade_{ bidTrade }
			, askTrade_{ askTrade }
		{ }
	
	//public api's	
	const TradeInfo& GetBidTrade() const { return bidTrade_; }
	const TradeInfo& GetAskTrade() const { return askTrade_; }
	
	//private api's
	private:
		TradeInfo bidTrade_;
		TradeInfo askTrade_;
		
};

//One order can consist of many trades => vector of Trade
using Trades = std::vector<Trade>;

//actual Orderbook
class Orderbook
{
	private:
		
		struct OrderEntry //Order entry object to have O(1) access to each element in bid/ask maps
		{
			OrderPointer order_{ nullptr };
			OrderPointers::iterator location_;
		};
		
		//ordered by price first then time 
		std::map<Price, OrderPointers, std::greater<Price>> bids_; //bids stored in descending order from best bid
		std::map<Price, OrderPointers, std::less<Price>> asks_; //asks stored in ascending order from best ask
		std::unordered_map<OrderId, OrderEntry> orders_;
		
		//if orderbook can match incoming order
		bool CanMatch(Side side, Price price) const
		{
			if (side == Side::Buy)
			{
				if (asks_.empty())
					return false;
					
				const auto& [bestAsk, _] = *asks_.begin(); //get best asks (lowest price) as first item from ordered map
				return price >= bestAsk;
			}
			else
			{
				if (bids_.empty())
					return false;
				const auto& [bestBid, _] = *bids_.begin(); //get best bid (highest price) as first item from ordered map
				return price <= bestBid;
			}
		}
		
		//Match function (check if best bid and best ask fit)
		Trades MatchOrders()
		{
			Trades trades;
			trades.reserve(orders_.size());
			
			while (true)
			{
				if (bids_.empty() || asks_.empty())
					break;
				
				auto& [bidPrice, bids] = *bids_.begin(); 
				auto& [askPrice, asks] = *asks_.begin(); 
				
				if (bidPrice < askPrice) //no match
					break;
				
				while (bids.size() && asks.size()) //match
				{
					auto& bid = bids.front(); //get best bid
					auto& ask = asks.front(); //get best ask
					
					Quantity quantity = std::min(bid->GetRemainingQuantity(), ask->GetRemainingQuantity()); //get trade quantity
					
					//fill order
					bid->Fill(quantity);
					ask->Fill(quantity);
					
					//erase bid/ask if they are filled
					if (bid->IsFilled())
					{
						bids.pop_front();
						orders_.erase(bid->GetOrderId());
					}
					
					if (ask->IsFilled())
					{
						asks.pop_front();
						orders_.erase(ask->GetOrderId());
					}
					
					//erase if bids/asks is empty
					if (bids.empty())
						bids_.erase(bidPrice);
						
					if (asks.empty())
						asks_.erase(askPrice);
					
					//collect trade info	
					trades.push_back(Trade{ 
						TradeInfo{ bid->GetOrderId(), bid->GetPrice(), quantity },
						TradeInfo{ ask->GetOrderId(), ask->GetPrice(), quantity }
						});
				}
			}
			
			//remove FillandKill orders
			if (!bids_.empty())
			{
				auto& [_, bids] = *bids_.begin();
				auto& order = bids.front();
				if (order->GetOrderType() == OrderType::FillAndKill)
					CancelOrder(order->GetOrderId());
			}
		
			if (!asks_.empty())
			{
				auto& [_, asks] = *asks_.begin();
				auto& order = asks.front();
				if (order->GetOrderType() == OrderType::FillAndKill)
					CancelOrder(order->GetOrderId());
			}
		
			return trades;
		}
		
	public:
		//Add orders
		Trades AddOrder(OrderPointer order)
		{	
			//only works if order id is unique
			if (orders_.contains(order->GetOrderId()))
				return { };
			//for fillandkill only works if we can match
			if (order->GetOrderType() == OrderType::FillAndKill && !CanMatch(order->GetSide(), order->GetPrice()))
				return { };
			
			OrderPointers::iterator iterator;
			
			//create a buy (bid)
			if (order->GetSide() == Side::Buy)
			{
				auto& orders = bids_[order->GetPrice()];
				orders.push_back(order);
				iterator = std::next(orders.begin(), orders.size() - 1);
			}
			//create a sell (ask)
			else
			{
				auto& orders = asks_[order->GetPrice()];
				orders.push_back(order);
				iterator = std::next(orders.begin(), orders.size() - 1);
			}
			
			//insert additional order
			orders_.insert({ order->GetOrderId(), OrderEntry{ order, iterator} });
			//run match
			return MatchOrders();
		}
		
		//remove order from orderbook
		void CancelOrder(OrderId orderId)
		{
			if (!orders_.contains(orderId))
				return;
			
			const auto& [order, iterator] = orders_.at(orderId);
			orders_.erase(orderId);
			
			//remove from sell side
			if (order->GetSide() == Side::Sell)
			{
				auto price = order->GetPrice();
				auto& orders = asks_.at(price);
				orders.erase(iterator);
				if (orders.empty())
					asks_.erase(price); //completely remove price level if order does not exist anymore at all
			}
			//remove from buy side
			else
			{
				auto price = order->GetPrice();
				auto& orders = bids_.at(price);
				orders.erase(iterator);
				if (orders.empty())
					bids_.erase(price); //completely remove price level if order does not exist anymore at all
			}
		}
		
		//modify (cancel order and add order)
		Trades MatchOrder(OrderModify order)
		{
			if (!orders_.contains(order.GetOrderId()))
				return { };
			
			const auto& [existingOrder, _] = orders_.at(order.GetOrderId()); //get order
			CancelOrder(order.GetOrderId()); //cancel the order
			return AddOrder(order.ToOrderPointer(existingOrder->GetOrderType())); //add order
		}
		
		std::size_t Size() const { return orders_.size(); } //check number of orders
		
		//get infos
		OrderbookLevelInfos GetOrderInfos() const
		{
			LevelInfos bidInfos, askInfos;
			bidInfos.reserve(orders_.size());
			askInfos.reserve(orders_.size());
			
			//lambda to get quantities for each price
			auto CreateLevelInfos = [](Price price, const OrderPointers& orders)
			{
				return LevelInfo{ price, std::accumulate(orders.begin(), orders.end(), (Quantity)0,
				[](Quantity runningSum, const OrderPointer& order)
				{ return runningSum + order->GetRemainingQuantity(); }) };
			};
			//create bid and ask infos
			for (const auto& [price, orders] : bids_)
				bidInfos.push_back(CreateLevelInfos(price, orders));
				
			for (const auto& [price, orders] : asks_)
				askInfos.push_back(CreateLevelInfos(price, orders));
				
			return OrderbookLevelInfos{ bidInfos, askInfos };
		}
};

int main()
{
	//std::cout << std::format("Hello World\n");
	Orderbook orderbook;
	const OrderId orderId = 1;
	orderbook.AddOrder(std::make_shared<Order>(OrderType::GoodTillCancel, orderId, Side::Buy, 100, 10));
	std::cout << orderbook.Size() << std::endl;
	orderbook.CancelOrder(orderId);
	std::cout << orderbook.Size() << std::endl;
	return 0;
}

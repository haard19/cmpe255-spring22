from tabnanny import verbose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        topx = self.chipo[:count]
        print(topx.to_markdown())
        
    def count(self) -> int:
        return self.chipo.shape[0]
    
    def info(self) -> None:
        print(self.chipo.info())
    
    def num_column(self) -> int:
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        print(self.chipo.columns)
    
    def most_ordered_item(self):
        item_name, order_id, quantity = self.chipo.groupby('item_name')['order_id','quantity'].agg('sum').sort_values('quantity', ascending=False).reset_index().loc[0]
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       return self.chipo.quantity.sum()
   
    def total_sales(self) -> float:
        self.chipo.item_price = self.chipo.item_price.apply(lambda x: float(x.strip().replace('$','')))
        self.chipo.loc[:, 'temp'] = self.chipo['item_price']*self.chipo['quantity']
        return self.chipo['temp'].sum()
   
    def num_orders(self) -> int:
        return self.chipo.order_id.nunique()
    
    def average_sales_amount_per_order(self) -> float:
        numOrders = Solution.num_orders(self)
        return round(self.chipo.temp.sum()/numOrders, 2)

    def num_different_items_sold(self) -> int:
        return self.chipo.item_name.nunique()
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        temp = pd.DataFrame.from_dict({k: v for k, v in sorted(letter_counter.items(), key=lambda item: item[1], reverse=True)}, orient='index').reset_index()

        temp.rename(columns={'index':'item_name', 0:'orders'}, inplace=True)

        ax = temp[:x].plot.bar('item_name', 'orders', rot=30)
        ax.set_xlabel('Items')
        ax.set_ylabel('Number of Orders')
        ax.set_title('Most popular items')
        plt.show(block=True)
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        plot_df = self.chipo.groupby('order_id')['quantity','item_price'].agg({'quantity': 'count', 'item_price': 'sum'}).reset_index(drop=True)

        ax = plot_df.plot.scatter('item_price', 'quantity', s=50)
        ax.set_xlabel('Order Price')
        ax.set_ylabel('Num Items')
        ax.set_title('Numer of items per order price')
        plt.show(block=True)
    
        
def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    assert quantity == 761
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    

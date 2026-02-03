using System.ComponentModel.DataAnnotations;

namespace VehicleCounting.Models
{
    public class Category
    {
        [Key]
        public int Id{ get; set; }
        public int Time { get; set; }
        public int VideoName { get; set; }
        public int Bus { get; set; }
        public int Jeep { get; set; }
        public int Tricycle { get; set; }
        public int Van { get; set; }
        public int Truck { get; set; }
        public int Car { get; set; }
    }
}

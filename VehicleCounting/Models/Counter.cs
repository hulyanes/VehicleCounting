using Microsoft.VisualBasic;
using System.ComponentModel.DataAnnotations;

namespace VehicleCounting.Models
{
    public class Counter
    {
        [Key]
        public int Id{ get; set; }
        public DateAndTime DateAndTime { get; set; }
        public string VideoName { get; set; }
        public int Bus { get; set; }
        public int Jeep { get; set; }
        public int Tricycle { get; set; }
        public int Van { get; set; }
        public int Truck { get; set; }
        public int Car { get; set; }
    }
}
